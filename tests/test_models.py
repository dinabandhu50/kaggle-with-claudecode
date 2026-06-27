import numpy as np
import pandas as pd
from src.models import cross_validate


def make_synthetic(n=100, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_cross_validate_returns_correct_shapes():
    X, y = make_synthetic()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    models, oof_preds, fold_scores = cross_validate(X, y, params, n_splits=3)
    assert len(models) == 3
    assert oof_preds.shape == (100,)
    assert len(fold_scores) == 3


def test_cross_validate_oof_preds_in_unit_interval():
    X, y = make_synthetic()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    _, oof_preds, _ = cross_validate(X, y, params, n_splits=3)
    assert np.all(oof_preds >= 0) and np.all(oof_preds <= 1)


def test_fold_scores_are_valid_auc():
    X, y = make_synthetic()
    params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}
    _, _, fold_scores = cross_validate(X, y, params, n_splits=3)
    assert all(0 <= s <= 1 for s in fold_scores)
