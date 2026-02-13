from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

@dataclass
class BankFeatureBuilder(BaseEstimator, TransformerMixin):
    """Feature engineering for Bank Marketing dataset.

    - Optionally drop leakage-prone column 'duration'
    - Numeric transforms:
        * balance_log1p_signed: sign(balance) * log1p(abs(balance))
        * pdays_is_999: indicator of 'no previous contact' (often pdays==999)
        * previous_is_zero: indicator
        * campaign_log1p: log1p(campaign)
    - Cyclic encoding for month if present (month_sin, month_cos)
    """
    drop_duration: bool = True

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Drop leakage feature if desired
        if self.drop_duration and "duration" in df.columns:
            df = df.drop(columns=["duration"])

        # balance transform
        if "balance" in df.columns:
            b = pd.to_numeric(df["balance"], errors="coerce")
            df["balance_log1p_signed"] = np.sign(b) * np.log1p(np.abs(b))

        # campaign transform
        if "campaign" in df.columns:
            c = pd.to_numeric(df["campaign"], errors="coerce")
            df["campaign_log1p"] = np.log1p(np.maximum(c, 0))

        # pdays indicator
        if "pdays" in df.columns:
            p = pd.to_numeric(df["pdays"], errors="coerce")
            df["pdays_is_999"] = (p == 999).astype(int)

        # previous indicator
        if "previous" in df.columns:
            pr = pd.to_numeric(df["previous"], errors="coerce")
            df["previous_is_zero"] = (pr.fillna(0) == 0).astype(int)

        # month cyclic
        if "month" in df.columns:
            m = df["month"].astype(str).str.lower().map(_MONTH_MAP)
            # if month already numeric, keep it
            if m.isna().all():
                m = pd.to_numeric(df["month"], errors="coerce")
            angle = 2 * np.pi * (m.fillna(1) - 1) / 12.0
            df["month_sin"] = np.sin(angle)
            df["month_cos"] = np.cos(angle)

        return df
