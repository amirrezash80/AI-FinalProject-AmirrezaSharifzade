from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_bank_marketing import load_raw, split_xy
from src.features.preprocess import build_preprocess_pipeline
from src.training.split import stratified_split
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import set_seed

from src.training.prepare_data import to_binary_target
from src.models.model_zoo import default_model_specs

from src.evaluation.metrics import compute_classification_report, report_to_dict
from src.evaluation.plots import save_confusion_matrix, save_roc_curve
from src.evaluation.feature_importance import topk_importances, save_barh


def _safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    raise RuntimeError("Model does not support probability or decision_function.")


def _get_feature_names(preprocess) -> list[str] | None:
    try:
        names = preprocess.named_steps["preprocess"].get_feature_names_out().tolist()
        return names
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop_duration", action="store_true")
    p.add_argument("--keep_duration", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    p.add_argument("--save_models", action="store_true", help="Save each trained model artifact (joblib)")
    args = p.parse_args()

    set_seed(args.seed)

    df = load_raw(args.raw_path)
    X, y_raw, used_target = split_xy(df, target_col=args.target_col)
    y, y_info = to_binary_target(y_raw)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Target has only one class after binarization. "
            "Check your target column or mapping (pos_label / numeric mapping)."
        )

    drop_duration = True
    if args.keep_duration:
        drop_duration = False
    if args.drop_duration:
        drop_duration = True

    split = stratified_split(X, pd.Series(y), seed=args.seed)

    preprocess = build_preprocess_pipeline(drop_duration=drop_duration)
    Xtr = preprocess.fit_transform(split.X_train)
    Xva = preprocess.transform(split.X_valid)
    Xte = preprocess.transform(split.X_test)

    feat_names = _get_feature_names(preprocess)

    specs = default_model_specs(args.seed, np.asarray(split.y_train))

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    model_dir = out_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_reports = {}

    for spec in specs:
        model = spec.estimator
        model.fit(Xtr, split.y_train)

        prob_va = _safe_predict_proba(model, Xva)
        pred_va = (prob_va >= 0.5).astype(int)
        rep_va = compute_classification_report(split.y_valid, pred_va, prob_va)

        prob_te = _safe_predict_proba(model, Xte)
        pred_te = (prob_te >= 0.5).astype(int)
        rep_te = compute_classification_report(split.y_test, pred_te, prob_te)

        rows.append({
            "model": spec.name,
            "valid_f1": rep_va.f1,
            "valid_auc": rep_va.roc_auc,
            "test_f1": rep_te.f1,
            "test_auc": rep_te.roc_auc,
            "test_acc": rep_te.accuracy,
        })

        all_reports[spec.name] = {
            "valid": report_to_dict(rep_va),
            "test": report_to_dict(rep_te),
        }

        save_confusion_matrix(split.y_valid, pred_va, fig_dir / f"{spec.name}_cm_valid.png", f"{spec.name} — CM (Valid)")
        save_roc_curve(split.y_valid, prob_va, fig_dir / f"{spec.name}_roc_valid.png", f"{spec.name} — ROC (Valid)")
        save_confusion_matrix(split.y_test, pred_te, fig_dir / f"{spec.name}_cm_test.png", f"{spec.name} — CM (Test)")
        save_roc_curve(split.y_test, prob_te, fig_dir / f"{spec.name}_roc_test.png", f"{spec.name} — ROC (Test)")

        if feat_names is not None:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(getattr(model, "feature_importances_"))
                items = topk_importances(imp, feat_names, k=20)
                save_barh(items, fig_dir / f"{spec.name}_feature_importance_top20.png", f"{spec.name} — Top-20 Feature Importance")
            elif spec.name == "LogReg" and hasattr(model, "coef_"):
                coef = np.asarray(model.coef_).reshape(-1)
                items = topk_importances(coef, feat_names, k=20)
                save_barh(items, fig_dir / f"{spec.name}_coef_top20.png", f"{spec.name} — Top-20 |coeff|")

        if args.save_models:
            joblib.dump({"preprocess": preprocess, "model": model}, model_dir / f"{spec.name}.joblib")

    df_cmp = pd.DataFrame(rows).sort_values(by=["valid_f1", "valid_auc"], ascending=False)
    df_cmp.to_csv(tab_dir / "phase2_model_comparison.csv", index=False)

    payload = {
        "seed": args.seed,
        "raw_target_col": used_target,
        "target_mapping": y_info.get("mapping"),
        "drop_duration": drop_duration,
        "models_included": [r["model"] for r in rows],
        "comparison_table": str(tab_dir / "phase2_model_comparison.csv"),
        "reports": all_reports,
        "note": "If XGBoost import fails on your environment, it will be skipped automatically.",
    }
    (tab_dir / "phase2_model_reports.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(df_cmp)
    print("\nSaved:")
    print("-", tab_dir / "phase2_model_comparison.csv")
    print("-", tab_dir / "phase2_model_reports.json")
    print("-", fig_dir)

if __name__ == "__main__":
    main()
