from __future__ import annotations
import numpy as np
import pandas as pd

def normalize_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """Treat 'unknown' (case-insensitive) as missing for object columns."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str)
            out[c] = s.where(~s.str.lower().eq("unknown"), other=np.nan)
    return out

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Light type normalization for known numeric-like columns if present."""
    out = df.copy()
    for c in ["age", "balance", "duration", "campaign", "pdays", "previous"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
