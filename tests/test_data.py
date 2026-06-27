import pandas as pd
from src.data import preprocess


def test_preprocess_train_splits_target():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "feat_a": [0.1, 0.2, 0.3],
        "feat_b": [10, 20, 30],
        "target": [0, 1, 0],
    })
    X, y = preprocess(df, target_col="target", drop_cols=["id"])
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert list(y) == [0, 1, 0]


def test_preprocess_test_returns_none_for_y():
    df = pd.DataFrame({
        "id": [1, 2],
        "feat_a": [0.1, 0.2],
        "feat_b": [10, 20],
    })
    X, y = preprocess(df, target_col="target", drop_cols=["id"])
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert y is None


def test_preprocess_drops_specified_cols():
    df = pd.DataFrame({
        "id": [1],
        "extra": [99],
        "feat": [0.5],
        "target": [1],
    })
    X, y = preprocess(df, target_col="target", drop_cols=["id", "extra"])
    assert "id" not in X.columns
    assert "extra" not in X.columns
