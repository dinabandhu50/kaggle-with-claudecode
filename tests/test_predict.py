import numpy as np
import pandas as pd
from src.predict import build_submission


def test_build_submission_shape():
    ids = pd.Series([101, 102, 103])
    preds = np.array([0.1, 0.9, 0.5])
    df = build_submission(ids, preds, target_col="target")
    assert list(df.columns) == ["id", "target"]
    assert len(df) == 3


def test_build_submission_values():
    ids = pd.Series([1, 2])
    preds = np.array([0.3, 0.7])
    df = build_submission(ids, preds, target_col="label")
    assert list(df["label"]) == [0.3, 0.7]
    assert list(df["id"]) == [1, 2]
