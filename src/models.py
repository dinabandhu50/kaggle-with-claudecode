import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
    eval_metric: str = "logloss",
    random_state: int = 42,
) -> tuple[list[xgb.XGBClassifier], np.ndarray, list[float]]:
    """Run stratified K-fold cross-validation with XGBoost.

    Returns (models, oof_preds, fold_scores) where models are saved for
    later ensemble averaging on the test set.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    fold_scores = []
    models = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            **params,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, oof_preds[val_idx]))
        models.append(model)

    return models, oof_preds, fold_scores
