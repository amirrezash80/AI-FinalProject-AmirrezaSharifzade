from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load_bank_marketing import load_raw, split_xy
from src.training.prepare_data import to_binary_target
from src.training.split import stratified_split
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import set_seed

from src.evaluation.metrics import compute_classification_report, report_to_dict
from src.evaluation.plots import save_confusion_matrix, save_roc_curve
from src.evaluation.threshold import sweep_thresholds, pick_best_threshold, rows_to_dicts
from src.evaluation.plots_extra import (
    save_f1_vs_threshold,
    save_precision_recall_curve,
    save_calibration_curve,
)

try:
    from sklearn.calibration import CalibratedClassifierCV
    _HAS_CAL = True
except Exception:
    CalibratedClassifierCV = None  # type: ignore
    _HAS_CAL = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop_duration", action="store_true")
    p.add_argument("--keep_duration", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    p.add_argument("--best_model_path", type=str, default=None, help="Default: results/models/phase2_best_model.joblib")
    p.add_argument("--metric", type=str, default="f1", choices=["f1", "accuracy", "precision", "recall"])
    p.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    p.add_argument("--n_bins", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    model_dir = out_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = Path(args.best_model_path) if args.best_model_path else (model_dir / "phase2_best_model.joblib")
    if not best_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_path}. Run tuning first.")

    df = load_raw(args.raw_path)
    X, y_raw, used_target = split_xy(df, target_col=args.target_col)
    y, y_info = to_binary_target(y_raw)

    drop_duration = True
    if args.keep_duration:
        drop_duration = False
    if args.drop_duration:
        drop_duration = True

    split = stratified_split(X, pd.Series(y), seed=args.seed)

    best_pipe = joblib.load(best_path)

    calibrated_pipe = None
    if args.calibrate != "none":
        if not _HAS_CAL:
            raise RuntimeError("Calibration module not available in your sklearn installation.")
        try:
            cal = CalibratedClassifierCV(estimator=best_pipe, method=args.calibrate, cv=3)
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=best_pipe, method=args.calibrate, cv=3)
        cal.fit(split.X_train, split.y_train)
        calibrated_pipe = cal
        joblib.dump(calibrated_pipe, model_dir / f"phase2_best_model_calibrated_{args.calibrate}.joblib")

    def _prob(pipe, Xs):
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(Xs)[:, 1]
        if hasattr(pipe, "decision_function"):
            s = pipe.decision_function(Xs)
            s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            return s
        raise RuntimeError("Pipeline does not support predict_proba/decision_function.")

    pipe_for_eval = calibrated_pipe if calibrated_pipe is not None else best_pipe

    prob_va = _prob(pipe_for_eval, split.X_valid)
    prob_te = _prob(pipe_for_eval, split.X_test)

    thresholds = np.linspace(0.05, 0.95, 91)
    rows = sweep_thresholds(split.y_valid, prob_va, thresholds)
    best = pick_best_threshold(rows, metric=args.metric)

    df_thr = pd.DataFrame(rows_to_dicts(rows))
    df_thr.to_csv(tab_dir / "phase2_threshold_sweep_valid.csv", index=False)

    save_f1_vs_threshold(df_thr["threshold"].to_numpy(), df_thr["f1"].to_numpy(),
                         fig_dir / "phase2_f1_vs_threshold_valid.png",
                         "Phase-2 — F1 vs Threshold (Valid)")
    save_precision_recall_curve(split.y_valid, prob_va, fig_dir / "phase2_pr_curve_valid.png",
                                "Phase-2 — Precision-Recall (Valid)")
    save_roc_curve(split.y_valid, prob_va, fig_dir / "phase2_roc_valid_bestpipe.png",
                   "Phase-2 — ROC (Valid)")
    save_calibration_curve(split.y_valid, prob_va, fig_dir / "phase2_calibration_curve_valid.png",
                           f"Phase-2 — Calibration (Valid) [{args.calibrate}]", n_bins=args.n_bins)

    def _eval_at(th, y_true, y_prob):
        y_pred = (y_prob >= th).astype(int)
        rep = compute_classification_report(y_true, y_pred, y_prob)
        return rep, y_pred

    rep_te_05, pred_te_05 = _eval_at(0.5, split.y_test, prob_te)
    rep_te_bt, pred_te_bt = _eval_at(best.threshold, split.y_test, prob_te)

    save_confusion_matrix(split.y_test, pred_te_05, fig_dir / "phase2_cm_test_threshold_0.5.png",
                          "Phase-2 — CM (Test) thr=0.5")
    save_confusion_matrix(split.y_test, pred_te_bt, fig_dir / f"phase2_cm_test_threshold_{best.threshold:.2f}.png",
                          f"Phase-2 — CM (Test) thr={best.threshold:.2f}")

    payload = {
        "seed": args.seed,
        "raw_target_col": used_target,
        "target_mapping": y_info.get("mapping"),
        "drop_duration": drop_duration,
        "calibration": args.calibrate,
        "selection_metric": args.metric,
        "best_threshold_valid": best.__dict__,
        "test_at_0.5": report_to_dict(rep_te_05),
        "test_at_best_threshold": report_to_dict(rep_te_bt),
        "artifacts": {
            "threshold_table": str(tab_dir / "phase2_threshold_sweep_valid.csv"),
            "plots_dir": str(fig_dir),
        },
    }
    (tab_dir / "phase2_threshold_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
