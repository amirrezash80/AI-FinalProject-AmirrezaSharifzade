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
from src.utils.paths import DATA_PROCESSED
from src.utils.seed import set_seed

def to_binary_target(y: pd.Series, pos_label: str | None = "yes") -> tuple[pd.Series, dict]:
    """Convert target to 0/1 robustly.

    - If values include yes/no -> yes=1
    - If values include true/false -> true=1
    - If numeric 0/1 -> keep
    - Else: pick the *rarer* class as positive (common in imbalanced settings)

    Returns (y_bin, info_dict)
    """
    info: dict = {}
    s = y.copy()

    # try numeric
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        uniq = sorted(s_num.unique().tolist())
        if set(uniq).issubset({0, 1}):
            info["mapping"] = "numeric(0/1)"
            return s_num.astype(int), info
        info["mapping"] = "numeric(>0 as positive)"
        return (s_num > 0).astype(int), info

    s_str = s.astype(str).str.strip().str.lower()
    uniq = set(s_str.unique().tolist())
    info["unique_labels"] = sorted(list(uniq))

    if uniq.issubset({"yes", "no"}):
        info["mapping"] = "yes->1"
        return (s_str == "yes").astype(int), info

    if uniq.issubset({"true", "false"}):
        info["mapping"] = "true->1"
        return (s_str == "true").astype(int), info

    if pos_label is not None and pos_label.lower() in uniq:
        info["mapping"] = f"{pos_label.lower()}->1"
        return (s_str == pos_label.lower()).astype(int), info

    # fallback: choose rarer class as positive
    vc = s_str.value_counts()
    pos = vc.index[-1]  # last is rarest
    info["mapping"] = f"rarer_class('{pos}')->1"
    return (s_str == pos).astype(int), info

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, default=None, help="Optional path to raw CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_col", type=str, default=None, help="Target column (default: auto-detect)")
    p.add_argument("--pos_label", type=str, default="yes", help="Positive label when target is string (default: yes)")
    p.add_argument("--drop_duration", action="store_true", help="Drop leakage-prone 'duration' feature (recommended)")
    p.add_argument("--keep_duration", action="store_true", help="Keep 'duration' for ablation experiment")
    p.add_argument("--save", action="store_true", help="Save processed arrays + preprocessor to data/processed")
    p.add_argument("--dense", action="store_true", help="Save dense arrays (otherwise sparse may be saved)")
    args = p.parse_args()

    set_seed(args.seed)
    df = load_raw(args.raw_path)
    X, y_raw, used_target = split_xy(df, target_col=args.target_col)

    y, y_info = to_binary_target(y_raw, pos_label=args.pos_label)

    drop_duration = True
    if args.keep_duration:
        drop_duration = False
    if args.drop_duration:
        drop_duration = True

    split = stratified_split(X, y, seed=args.seed)
    pipe = build_preprocess_pipeline(drop_duration=drop_duration)

    Xtr = pipe.fit_transform(split.X_train)
    Xva = pipe.transform(split.X_valid)
    Xte = pipe.transform(split.X_test)

    feat_names = None
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        feat_names = None

    summary = {
        "seed": args.seed,
        "raw_target_col": used_target,
        "target_mapping": y_info.get("mapping"),
        "unique_labels": y_info.get("unique_labels"),
        "drop_duration": drop_duration,
        "n_train": int(len(split.y_train)),
        "n_valid": int(len(split.y_valid)),
        "n_test": int(len(split.y_test)),
        "pos_rate_train": float(np.mean(split.y_train)),
        "pos_rate_valid": float(np.mean(split.y_valid)),
        "pos_rate_test": float(np.mean(split.y_test)),
        "X_train_shape": list(Xtr.shape),
        "X_valid_shape": list(Xva.shape),
        "X_test_shape": list(Xte.shape),
        "has_feature_names": feat_names is not None,
    }

    print(json.dumps(summary, indent=2))

    if args.save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out_dir = Path(DATA_PROCESSED)
        joblib.dump(pipe, out_dir / "preprocess.joblib")

        if args.dense:
            Xtr_s = np.asarray(Xtr.todense() if hasattr(Xtr, "todense") else Xtr)
            Xva_s = np.asarray(Xva.todense() if hasattr(Xva, "todense") else Xva)
            Xte_s = np.asarray(Xte.todense() if hasattr(Xte, "todense") else Xte)
            np.savez_compressed(
                out_dir / "dataset_dense.npz",
                X_train=Xtr_s, X_valid=Xva_s, X_test=Xte_s,
                y_train=split.y_train.to_numpy(),
                y_valid=split.y_valid.to_numpy(),
                y_test=split.y_test.to_numpy(),
            )
        else:
            joblib.dump(
                {
                    "X_train": Xtr,
                    "X_valid": Xva,
                    "X_test": Xte,
                    "y_train": split.y_train.to_numpy(),
                    "y_valid": split.y_valid.to_numpy(),
                    "y_test": split.y_test.to_numpy(),
                },
                out_dir / "dataset_sparse.joblib",
            )

        if feat_names is not None:
            (out_dir / "feature_names.json").write_text(json.dumps(feat_names, indent=2), encoding="utf-8")

        (out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Saved processed artifacts to:", out_dir)

if __name__ == "__main__":
    main()
