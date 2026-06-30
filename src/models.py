import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _make_xgb(params: dict, eval_metric: str, early_stopping_rounds: int):
    return xgb.XGBClassifier(
        **params,
        device="cuda",
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
    )


def _fit_xgb(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


def _make_lgb(params: dict, early_stopping_rounds: int):
    return lgb.LGBMClassifier(
        **params,
        device="cuda",
        verbose=-1,
    )


def _fit_lgb(model, X_train, y_train, X_val, y_val, early_stopping_rounds: int):
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(-1),
    ]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks,
    )


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
    eval_metric: str = "logloss",
    random_state: int = 42,
    model_type: str = "xgboost",
) -> tuple[list, np.ndarray, list[float]]:
    """Run stratified K-fold cross-validation with XGBoost or LightGBM.

    Returns (models, oof_preds, fold_scores).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    models = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == "lightgbm":
            model = _make_lgb(params, early_stopping_rounds)
            _fit_lgb(model, X_train, y_train, X_val, y_val, early_stopping_rounds)
        else:  # xgboost (default)
            model = _make_xgb(params, eval_metric, early_stopping_rounds)
            _fit_xgb(model, X_train, y_train, X_val, y_val)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, oof_preds[val_idx]))
        models.append(model)

    return models, oof_preds, fold_scores
