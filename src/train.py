import sys
from pathlib import Path

import wandb
import yaml
from sklearn.metrics import roc_auc_score

from src.data import load_train, load_test, preprocess
from src.features import get_features
from src.models import cross_validate
from src.predict import generate_submission


def run(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["experiment_name"],
        config=cfg,
    )

    # Load and preprocess
    train_df = load_train()
    X, y = preprocess(train_df, cfg["data"]["target_col"], cfg["data"]["drop_cols"])
    X = get_features(X)

    # Train
    models, oof_preds, fold_scores = cross_validate(
        X, y, cfg["hyperparams"], n_splits=cfg["cv"]["n_splits"]
    )

    # Log to W&B
    for i, score in enumerate(fold_scores):
        wandb.log({f"fold_{i + 1}_auc": score})
    oof_score = roc_auc_score(y, oof_preds)
    wandb.log({"oof_auc": oof_score})
    print(f"OOF AUC: {oof_score:.5f}")

    # Generate submission
    test_df = load_test()
    ids = test_df["id"]
    X_test, _ = preprocess(test_df, cfg["data"]["target_col"], cfg["data"]["drop_cols"])
    X_test = get_features(X_test)

    path = generate_submission(
        models, X_test, ids, cfg["data"]["target_col"], cfg["experiment_name"]
    )
    wandb.save(str(path))
    wandb.finish()


if __name__ == "__main__":
    run(sys.argv[1])
