from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "raw" / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "raw" / "test.csv")


def preprocess(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if target_col in df.columns:
        y = df[target_col].reset_index(drop=True)
        X = df.drop(columns=cols_to_drop + [target_col])
    else:
        y = None
        X = df.drop(columns=cols_to_drop)
    return X.reset_index(drop=True), y
