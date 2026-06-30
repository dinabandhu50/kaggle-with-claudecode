from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_train() -> pd.DataFrame:
    """Load raw training CSV from data/raw/train.csv."""
    return pd.read_csv(DATA_DIR / "raw" / "train.csv")


def load_test() -> pd.DataFrame:
    """Load raw test CSV from data/raw/test.csv."""
    return pd.read_csv(DATA_DIR / "raw" / "test.csv")


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace spaces in column names with underscores for XGBoost compatibility."""
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split df into features X and target y, dropping id/metadata columns.

    String targets are label-encoded to integers via sorted unique values
    (alphabetical order: e.g. Absence=0, Presence=1).
    Returns (X, None) for test data where target_col is absent.
    """
    df = _sanitize_columns(df.copy())
    # Sanitize target_col and drop_cols names to match
    target_col = target_col.replace(" ", "_")
    drop_cols = [c.replace(" ", "_") for c in drop_cols]

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if target_col in df.columns:
        y = df[target_col].reset_index(drop=True)
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(
                pd.Categorical(y, categories=sorted(y.unique())).codes,
                dtype=int,
            )
        X = df.drop(columns=cols_to_drop + [target_col])
    else:
        y = None
        X = df.drop(columns=cols_to_drop)
    return X.reset_index(drop=True), y
