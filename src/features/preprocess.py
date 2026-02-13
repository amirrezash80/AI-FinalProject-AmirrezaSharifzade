from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.features.cleaning import normalize_unknowns, ensure_types
from src.features.feature_builder import BankFeatureBuilder

def clean_df(X):
    """Top-level function (pickleable) used by FunctionTransformer."""
    if isinstance(X, pd.DataFrame):
        X2 = ensure_types(X)
        X2 = normalize_unknowns(X2)
        return X2
    X2 = pd.DataFrame(X)
    X2 = ensure_types(X2)
    X2 = normalize_unknowns(X2)
    return X2

def build_preprocess_pipeline(drop_duration: bool = True) -> Pipeline:
    """Full preprocessing pipeline:
    Cleaning -> Feature Engineering -> ColumnTransformer(num/cat)
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, selector(dtype_include=np.number)),
            ("cat", categorical_pipe, selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[
        ("clean", FunctionTransformer(clean_df, validate=False)),
        ("feat", BankFeatureBuilder(drop_duration=drop_duration)),
        ("preprocess", pre),
    ])
    return pipe
