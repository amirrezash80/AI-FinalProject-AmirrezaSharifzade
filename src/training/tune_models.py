from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.load_bank_marketing import load_raw, split_xy
from src.features.preprocess import build_preprocess_pipeline
from src.training.split import stratified_split
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import set_seed
from src.training.prepare_data import to_binary_target
from src.evaluation.metrics import compute_classification_report, report_to_dict

# Optional distributions
try:
    from scipy.stats import loguniform, randint  # type: ignore
    _HAS_SCIPY = True
except Exception:
    loguniform = None  # type: ignore
    randint = None  # type: ignore
    _HAS_SCIPY = False

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False


def _safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    raise RuntimeError("Model does not support probability or decision_function.")


def build_search_space(seed: int, y_train: np.ndarray) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Returns list of (name, estimator, param_distributions) for tuning."""
    spaces: List[Tuple[str, Any, Dict[str, Any]]] = []

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    if _HAS_SCIPY:
        lr_params = {"model__C": loguniform(1e-3, 1e2)}  # type: ignore
    else:
        lr_params = {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    spaces.append(("LogReg", lr, lr_params))

    # SVM (RBF)
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    if _HAS_SCIPY:
        svm_params = {
            "model__C": loguniform(1e-2, 1e2),  # type: ignore
            "model__gamma": loguniform(1e-4, 1e0),  # type: ignore
        }
    else:
        svm_params = {
            "model__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
            "model__gamma": [1e-4, 1e-3, 1e-2, 1e-1, "scale"],
        }
    spaces.append(("SVM-RBF", svm, svm_params))

    # KNN
    knn = KNeighborsClassifier()
    if _HAS_SCIPY:
        knn_params = {
            "model__n_neighbors": randint(3, 51),  # type: ignore
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    else:
        knn_params = {
            "model__n_neighbors": list(range(3, 51, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    spaces.append(("KNN", knn, knn_params))

    # RandomForest
    rf = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    if _HAS_SCIPY:
        rf_params = {
            "model__n_estimators": randint(200, 1000),  # type: ignore
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": randint(2, 21),  # type: ignore
            "model__min_samples_leaf": randint(1, 11),  # type: ignore
            "model__max_features": ["sqrt", "log2", None],
        }
    else:
        rf_params = {
            "model__n_estimators": [200, 400, 600, 800, 1000],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        }
    spaces.append(("RandomForest", rf, rf_params))

    # XGBoost (optional)
    if _HAS_XGB:
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        spw = (neg / pos) if pos > 0 else 1.0
        xgb = XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=float(spw),
        )
        if _HAS_SCIPY:
            xgb_params = {
                "model__n_estimators": randint(300, 1200),  # type: ignore
                "model__max_depth": randint(3, 9),  # type: ignore
                "model__learning_rate": loguniform(1e-2, 2e-1),  # type: ignore
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "model__reg_lambda": loguniform(1e-2, 10.0),  # type: ignore
            }
        else:
            xgb_params = {
                "model__n_estimators": [300, 600, 900, 1200],
                "model__max_depth": [3, 4, 5, 6, 7, 8],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "model__reg_lambda": [0.01, 0.1, 1.0, 3.0, 10.0],
            }
        spaces.append(("XGBoost", xgb, xgb_params))

    return spaces


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_path", type=str, default=None)
    p.add_argument("--target_col", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop_duration", action="store_true")
    p.add_argument("--keep_duration", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    p.add_argument("--n_iter", type=int, default=30, help="Random search iterations per model")
    p.add_argument("--cv", type=int, default=3, help="StratifiedKFold splits")
    p.add_argument("--scoring", type=str, default="f1", help="Scoring for selection (f1 recommended)")
    args = p.parse_args()

    set_seed(args.seed)

    df = load_raw(args.raw_path)
    X, y_raw, used_target = split_xy(df, target_col=args.target_col)
    y, y_info = to_binary_target(y_raw)

    if len(np.unique(y)) < 2:
        raise ValueError("Target has only one class after binarization. Check target_col/mapping.")

    drop_duration = True
    if args.keep_duration:
        drop_duration = False
    if args.drop_duration:
        drop_duration = True

    split = stratified_split(X, pd.Series(y), seed=args.seed)

    # We tune using CV only on TRAIN (avoid leaking valid/test)
    X_train = split.X_train
    y_train = np.asarray(split.y_train)
    X_valid = split.X_valid
    y_valid = np.asarray(split.y_valid)
    X_test = split.X_test
    y_test = np.asarray(split.y_test)

    preprocess = build_preprocess_pipeline(drop_duration=drop_duration)

    spaces = build_search_space(args.seed, y_train)

    out_dir = Path(args.out_dir)
    tab_dir = out_dir / "tables"
    model_dir = out_dir / "models"
    tab_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    all_rows = []
    best_overall = None  # (best_score, name, search_obj)

    for name, estimator, param_dist in spaces:
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring=args.scoring,
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_score = float(search.best_score_)
        best_params = search.best_params_

        # Evaluate on valid/test with best estimator
        best_est = search.best_estimator_
        prob_va = _safe_predict_proba(best_est.named_steps["model"], best_est.named_steps["preprocess"].transform(X_valid))
        pred_va = (prob_va >= 0.5).astype(int)
        rep_va = compute_classification_report(y_valid, pred_va, prob_va)

        prob_te = _safe_predict_proba(best_est.named_steps["model"], best_est.named_steps["preprocess"].transform(X_test))
        pred_te = (prob_te >= 0.5).astype(int)
        rep_te = compute_classification_report(y_test, pred_te, prob_te)

        row = {
            "model": name,
            "cv_best_score": best_score,
            "scoring": args.scoring,
            "valid_f1": rep_va.f1,
            "valid_auc": rep_va.roc_auc,
            "test_f1": rep_te.f1,
            "test_auc": rep_te.roc_auc,
            "test_acc": rep_te.accuracy,
            "best_params": json.dumps(best_params),
        }
        all_rows.append(row)

        if best_overall is None or best_score > best_overall[0]:
            best_overall = (best_score, name, search)

    df_res = pd.DataFrame(all_rows).sort_values(by=["cv_best_score"], ascending=False)
    df_res.to_csv(tab_dir / "phase2_tuning_results.csv", index=False)

    assert best_overall is not None
    best_score, best_name, best_search = best_overall
    best_est = best_search.best_estimator_

    # Final evaluation and save artifact
    # Use the same evaluation as above but compute again for consistency
    Xva_t = best_est.named_steps["preprocess"].transform(X_valid)
    Xte_t = best_est.named_steps["preprocess"].transform(X_test)
    model = best_est.named_steps["model"]

    prob_va = _safe_predict_proba(model, Xva_t)
    pred_va = (prob_va >= 0.5).astype(int)
    rep_va = compute_classification_report(y_valid, pred_va, prob_va)

    prob_te = _safe_predict_proba(model, Xte_t)
    pred_te = (prob_te >= 0.5).astype(int)
    rep_te = compute_classification_report(y_test, pred_te, prob_te)

    payload = {
        "seed": args.seed,
        "raw_target_col": used_target,
        "target_mapping": y_info.get("mapping"),
        "drop_duration": drop_duration,
        "selection_metric": args.scoring,
        "best_model": best_name,
        "best_cv_score": best_score,
        "best_params": best_search.best_params_,
        "valid": report_to_dict(rep_va),
        "test": report_to_dict(rep_te),
        "notes": {
            "cv": args.cv,
            "n_iter_per_model": args.n_iter,
            "xgboost_included": _HAS_XGB,
            "scipy_distributions": _HAS_SCIPY,
        },
    }
    (tab_dir / "phase2_best_model_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    joblib.dump(best_est, model_dir / "phase2_best_model.joblib")

    print(df_res)
    print("\nBest model:", best_name, "CV score:", best_score)
    print("Saved:")
    print("-", tab_dir / "phase2_tuning_results.csv")
    print("-", tab_dir / "phase2_best_model_report.json")
    print("-", model_dir / "phase2_best_model.joblib")


if __name__ == "__main__":
    main()
