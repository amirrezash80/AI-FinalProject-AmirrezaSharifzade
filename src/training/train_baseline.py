from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.data.load_bank_marketing import load_raw, split_xy
from src.features.preprocess import build_preprocess_pipeline
from src.training.split import stratified_split
from src.models.baselines import BaselineConfig, build_logreg
from src.evaluation.metrics import compute_classification_report, report_to_dict
from src.evaluation.plots import save_confusion_matrix, save_roc_curve
from src.utils.paths import PROJECT_ROOT, DATA_PROCESSED
from src.utils.seed import set_seed

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop_duration", action="store_true")
    p.add_argument("--keep_duration", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    args = p.parse_args()

    set_seed(args.seed)

    df = load_raw(args.raw_path)
    X, y_raw, used_target = split_xy(df, target_col=args.target_col)

    # Import the same function from prepare_data to ensure consistent mapping
    from src.training.prepare_data import to_binary_target
    y, y_info = to_binary_target(y_raw)

    drop_duration = True
    if args.keep_duration:
        drop_duration = False
    if args.drop_duration:
        drop_duration = True

    split = stratified_split(X, y, seed=args.seed)

    preprocess = build_preprocess_pipeline(drop_duration=drop_duration)
    Xtr = preprocess.fit_transform(split.X_train)
    Xva = preprocess.transform(split.X_valid)
    Xte = preprocess.transform(split.X_test)

    model = build_logreg(BaselineConfig(seed=args.seed))
    model.fit(Xtr, split.y_train)

    # Evaluate on valid and test
    def _eval(name, Xs, ys):
        prob = model.predict_proba(Xs)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rep = compute_classification_report(ys, pred, prob)
        return rep, prob, pred

    rep_va, prob_va, pred_va = _eval("valid", Xva, split.y_valid)
    rep_te, prob_te, pred_te = _eval("test", Xte, split.y_test)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    model_dir = out_dir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save plots
    save_confusion_matrix(split.y_valid, pred_va, fig_dir / "baseline_logreg_cm_valid.png", "Baseline LogReg — CM (Valid)")
    save_roc_curve(split.y_valid, prob_va, fig_dir / "baseline_logreg_roc_valid.png", "Baseline LogReg — ROC (Valid)")
    save_confusion_matrix(split.y_test, pred_te, fig_dir / "baseline_logreg_cm_test.png", "Baseline LogReg — CM (Test)")
    save_roc_curve(split.y_test, prob_te, fig_dir / "baseline_logreg_roc_test.png", "Baseline LogReg — ROC (Test)")

    payload = {
        "dataset": "tabular",
        "raw_target_col": used_target,
        "target_mapping": y_info.get("mapping"),
        "drop_duration": drop_duration,
        "valid": report_to_dict(rep_va),
        "test": report_to_dict(rep_te),
    }

    (tab_dir / "baseline_logreg_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Save model + preprocess (pickleable)
    joblib.dump({"preprocess": preprocess, "model": model}, model_dir / "baseline_logreg.joblib")

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
