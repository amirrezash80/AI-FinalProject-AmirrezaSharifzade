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

def to_binary_target(y: pd.Series) -> pd.Series:
    y = y.astype(str).str.lower().str.strip()
    # OpenML/UCI uses yes/no
    return (y == "yes").astype(int)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, default=None, help="Optional path to raw CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop_duration", action="store_true", help="Drop leakage-prone 'duration' feature (recommended)")
    p.add_argument("--keep_duration", action="store_true", help="Keep 'duration' for ablation experiment")
    p.add_argument("--save", action="store_true", help="Save processed arrays + preprocessor to data/processed")
    p.add_argument("--dense", action="store_true", help="Save dense arrays (otherwise sparse may be saved)")
    args = p.parse_args()

    set_seed(args.seed)
    df = load_raw(args.raw_path)
    X, y_raw = split_xy(df, target_col="y")
    y = to_binary_target(y_raw)

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

    # Attempt to extract feature names
    feat_names = None
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()
    except Exception:
        feat_names = None

    summary = {
        "seed": args.seed,
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
            np.savez_compressed(out_dir / "dataset_dense.npz",
                                X_train=Xtr_s, X_valid=Xva_s, X_test=Xte_s,
                                y_train=split.y_train.to_numpy(),
                                y_valid=split.y_valid.to_numpy(),
                                y_test=split.y_test.to_numpy())
        else:
            # Save sparse safely via joblib
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
