from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_bank_marketing import load_raw, split_xy

DEFAULT_MODEL_PATH = Path("results/models/phase2_best_model.joblib")
DEFAULT_THRESHOLD_PATH = Path("results/tables/phase2_threshold_summary.json")


def load_threshold(path: Path):
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    try:
        return float(info.get("best_threshold_valid", {}).get("threshold"))
    except Exception:
        return None


def infer_expected_columns() -> list[str]:
    df = load_raw()
    X, _, _ = split_xy(df, target_col=None)
    return list(X.columns)


def get_prob(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    raise RuntimeError("Model does not support predict_proba/decision_function.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_csv", type=str, default="predictions.csv")
    p.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--threshold_path", type=str, default=str(DEFAULT_THRESHOLD_PATH))
    args = p.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    thr = load_threshold(Path(args.threshold_path))

    expected_cols = infer_expected_columns()

    df_in = pd.read_csv(args.input_csv)
    try:
        X_in, _, _ = split_xy(df_in, target_col=None)
    except Exception:
        X_in = df_in.copy()

    missing = [c for c in expected_cols if c not in X_in.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    X = X_in[expected_cols].copy()

    prob = get_prob(model, X)
    out = df_in.copy()
    out["probability"] = prob
    out["pred@0.5"] = (prob >= 0.5).astype(int)
    if thr is not None:
        out[f"pred@{thr:.2f}"] = (prob >= thr).astype(int)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
