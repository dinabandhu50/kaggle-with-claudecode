from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"


def build_submission(
    ids: pd.Series,
    preds: np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    """Build a submission DataFrame with id and prediction columns."""
    return pd.DataFrame({"id": ids, target_col: preds})


def generate_submission(
    models: list,
    X_test: pd.DataFrame,
    ids: pd.Series,
    target_col: str,
    experiment_name: str,
) -> Path:
    """Average fold model predictions and write timestamped submission CSV."""
    preds = np.mean(
        [m.predict_proba(X_test)[:, 1] for m in models], axis=0
    )
    submission = build_submission(ids, preds, target_col)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"{timestamp}_{experiment_name}.csv"
    submission.to_csv(path, index=False)
    print(f"Submission saved → {path}")
    return path
