from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.utils.paths import DATA_RAW

RAW_DEFAULT = DATA_RAW / "bank_marketing.csv"

def load_raw(path: str | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else RAW_DEFAULT
    if not p.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {p}. Run: python -m src.data.download_bank_marketing"
        )
    return pd.read_csv(p)

def split_xy(df: pd.DataFrame, target_col: str = "y"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
