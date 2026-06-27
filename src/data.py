from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_train() -> pd.DataFrame:
    """Load raw training CSV from data/raw/train.csv."""
    return pd.read_csv(DATA_DIR / "raw" / "train.csv")


def load_test() -> pd.DataFrame:
    """Load raw test CSV from data/raw/test.csv."""
    return pd.read_csv(DATA_DIR / "raw" / "test.csv")


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split df into features X and target y, dropping id/metadata columns.

    Returns (X, None) for test data where target_col is absent.
    """
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if target_col in df.columns:
        y = df[target_col].reset_index(drop=True)
        X = df.drop(columns=cols_to_drop + [target_col])
    else:
        y = None
        X = df.drop(columns=cols_to_drop)
    return X.reset_index(drop=True), y
