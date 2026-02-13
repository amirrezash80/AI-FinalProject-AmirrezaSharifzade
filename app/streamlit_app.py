from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.load_bank_marketing import load_raw, split_xy

DEFAULT_MODEL_PATH = Path("results/models/phase2_best_model.joblib")
DEFAULT_THRESHOLD_PATH = Path("results/tables/phase2_threshold_summary.json")


def load_threshold_info(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_expected_feature_columns() -> list[str]:
    # We infer expected columns from the raw dataset schema.
    # The trained pipeline expects a DataFrame with the same columns it was fitted on.
    df = load_raw()
    X, _, _ = split_xy(df, target_col=None)
    return list(X.columns)


def predict_df(model, df_features: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_features)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(df_features)
        prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        raise RuntimeError("Model does not support predict_proba/decision_function.")

    pred_05 = (prob >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "probability": prob,
            "pred@0.5": pred_05,
        }
    )


def main():
    st.set_page_config(page_title="Classic ML Demo", layout="wide")

    st.title("Demo — Classic ML (Phase 2)")
    st.write(
        "این دمو برای ارائه‌ی پروژه‌ست: یک فایل CSV با ستون‌های دیتاست می‌گیریم، "
        "با بهترین مدل آموزش‌دیده (Pipeline کامل) پیش‌بینی می‌کنیم و خروجی می‌دیم."
    )

    with st.sidebar:
        st.header("Paths")
        model_path = Path(st.text_input("Best model path", str(DEFAULT_MODEL_PATH)))
        thr_path = Path(st.text_input("Threshold summary path", str(DEFAULT_THRESHOLD_PATH)))
        st.caption("این مسیرها بعد از اجرای tuning و threshold ساخته می‌شن.")

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)
    thr_info = load_threshold_info(thr_path)

    best_thr = None
    try:
        best_thr = float(thr_info.get("best_threshold_valid", {}).get("threshold"))
    except Exception:
        best_thr = None

    if best_thr is not None:
        st.success(f"Best threshold (from VALID): {best_thr:.2f}")
    else:
        st.info("Threshold summary پیدا نشد یا threshold استخراج نشد؛ فقط pred@0.5 نمایش داده می‌شود.")

    st.subheader("1) Upload CSV")
    st.write("CSV باید همان ستون‌های دیتاست را داشته باشد (وجود ستون هدف هم مشکلی ندارد؛ خودکار حذف می‌شود).")

    expected_cols = get_expected_feature_columns()

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.code(
            "برای اجرا:\n"
            "streamlit run app/streamlit_app.py\n\n"
            "و یک CSV با ستون‌های دیتاست آپلود کن."
        )
        return

    df_in = pd.read_csv(up)

    # Drop target if present (auto-detect using split_xy)
    try:
        X_in, _, used_target = split_xy(df_in, target_col=None)
    except Exception:
        X_in = df_in.copy()
        used_target = None

    st.write("Input shape:", df_in.shape)
    if used_target is not None:
        st.caption(f"Detected target column in uploaded CSV and removed it: `{used_target}`")

    # Align columns with expected training schema
    missing = [c for c in expected_cols if c not in X_in.columns]
    extra = [c for c in X_in.columns if c not in expected_cols]

    col1, col2 = st.columns(2)
    with col1:
        if missing:
            st.error(f"Missing columns ({len(missing)}): {missing[:10]}{' ...' if len(missing)>10 else ''}")
        else:
            st.success("No missing columns ✅")
    with col2:
        if extra:
            st.warning(f"Extra columns will be ignored ({len(extra)}): {extra[:10]}{' ...' if len(extra)>10 else ''}")
        else:
            st.success("No extra columns ✅")

    if missing:
        st.stop()

    X_aligned = X_in[expected_cols].copy()

    st.subheader("2) Predictions")
    res = predict_df(model, X_aligned)

    if best_thr is not None:
        res[f"pred@{best_thr:.2f}"] = (res["probability"].to_numpy() >= best_thr).astype(int)

    out = pd.concat([X_in.reset_index(drop=True), res], axis=1)
    st.dataframe(out.head(50), use_container_width=True)

    st.subheader("3) Download predictions")
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()
