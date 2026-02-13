from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class Split:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    test_size: float = 0.2,
    valid_size: float = 0.2,
) -> Split:
    """Stratified split into train/valid/test.
    valid_size is fraction of total data (not of train).
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    valid_ratio = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tmp, y_tmp, test_size=valid_ratio, random_state=seed, stratify=y_tmp
    )
    return Split(X_train, X_valid, X_test, y_train, y_valid, y_test)
