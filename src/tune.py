"""Optuna hyperparameter tuning for XGBoost or LightGBM.

Usage:
    uv run python src/tune.py configs/xgb_hp_v1.yaml
    uv run python src/tune.py configs/lgb_hp_v1.yaml

The best params are written back to a <experiment_name>_best.yaml
file in configs/ and the final model is trained on full 5-fold CV.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import wandb
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_train, load_test, preprocess
from src.features import get_features, get_features_v2
from src.models import cross_validate, _fit_lgb, _fit_xgb
from src.predict import generate_submission
from src.train import save_oof

optuna.logging.set_verbosity(optuna.logging.WARNING)

_FEATURE_FNS = {
    "v1": get_features,
    "v2": get_features_v2,
}


def _xgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": 5000,   # high ceiling — early stopping decides actual count
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
    }


def _lgb_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": 5000,   # high ceiling — early stopping decides actual count
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
    }


def _objective(
    trial: optuna.Trial,
    X, y,
    n_splits: int,
    model_type: str,
    subsample_rows: int | None = None,
) -> float:
    params = _xgb_params(trial) if model_type == "xgboost" else _lgb_params(trial)

    # Subsample rows for speed during search
    if subsample_rows and len(X) > subsample_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), subsample_rows, replace=False)
        X_s = X.iloc[idx].reset_index(drop=True)
        y_s = y.iloc[idx].reset_index(drop=True)
    else:
        X_s, y_s = X, y

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y_s))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_s, y_s)):
        X_tr, X_val = X_s.iloc[train_idx], X_s.iloc[val_idx]
        y_tr, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]

        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(**params, device="cuda", verbose=-1)
            _fit_lgb(model, X_tr, y_tr, X_val, y_val, early_stopping_rounds=50)
        else:
            model = xgb.XGBClassifier(
                **params, device="cuda",
                eval_metric="logloss", early_stopping_rounds=50,
            )
            _fit_xgb(model, X_tr, y_tr, X_val, y_val)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_score = roc_auc_score(y_val, oof[val_idx])
        trial.report(fold_score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return roc_auc_score(y_s, oof)


def run(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    n_trials   = cfg.get("optuna", {}).get("n_trials", 50)
    tune_folds = cfg.get("optuna", {}).get("tune_folds", 3)
    tune_rows  = cfg.get("optuna", {}).get("tune_rows", 300_000)
    n_splits   = cfg["cv"]["n_splits"]
    model_type = cfg.get("model", "xgboost")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["experiment_name"],
        config=cfg,
    )

    train_df = load_train()
    X, y = preprocess(train_df, cfg["data"]["target_col"], cfg["data"]["drop_cols"])
    feature_fn = _FEATURE_FNS.get(cfg.get("features", "v2"), get_features_v2)
    X = feature_fn(X)
    print(f"Model: {model_type} | Features: {X.shape}")

    print(f"Running Optuna ({n_trials} trials, {tune_folds}-fold CV on {tune_rows} rows)…")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: _objective(trial, X, y, tune_folds, model_type, subsample_rows=tune_rows),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_value  = study.best_value
    print(f"\nBest OOF AUC (search): {best_value:.5f}")
    print(f"Best params: {best_params}")
    wandb.log({"best_oof_auc": best_value, **{f"best_{k}": v for k, v in best_params.items()}})

    # Persist best config
    best_cfg = dict(cfg)
    best_cfg["experiment_name"] = cfg["experiment_name"] + "_best"
    best_cfg["hyperparams"] = {**best_params, "random_state": 42}
    best_cfg_path = Path(config_path).parent / f"{cfg['experiment_name']}_best.yaml"
    best_cfg_path.write_text(yaml.dump(best_cfg, default_flow_style=False))
    print(f"Best config saved -> {best_cfg_path}")

    # Final full-CV run
    print(f"\nTraining final {model_type} model ({n_splits}-fold, full data)…")
    models, oof_preds, fold_scores = cross_validate(
        X, y, best_params, n_splits=n_splits, model_type=model_type,
    )
    for i, score in enumerate(fold_scores):
        wandb.log({f"fold_{i + 1}_auc": score})
    oof_score = roc_auc_score(y, oof_preds)
    wandb.log({"oof_auc": oof_score})
    print(f"Final OOF AUC: {oof_score:.5f}")

    train_ids = train_df["id"].values
    oof_path = save_oof(oof_preds, train_ids, cfg["experiment_name"])
    wandb.save(str(oof_path))

    test_df = load_test()
    ids = test_df["id"]
    X_test, _ = preprocess(test_df, cfg["data"]["target_col"], cfg["data"]["drop_cols"])
    X_test = feature_fn(X_test)
    path = generate_submission(
        models, X_test, ids, cfg["data"]["target_col"], cfg["experiment_name"]
    )
    wandb.save(str(path))
    wandb.finish()


if __name__ == "__main__":
    run(sys.argv[1])
