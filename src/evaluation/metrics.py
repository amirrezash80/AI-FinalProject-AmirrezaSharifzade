from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    cm: np.ndarray

def compute_classification_report(y_true, y_pred, y_prob=None) -> ClassificationReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # roc_auc_score requires both classes present
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_prob))

    return ClassificationReport(acc, prec, rec, f1, auc, cm)

def report_to_dict(r: ClassificationReport) -> Dict[str, Any]:
    return {
        "accuracy": r.accuracy,
        "precision": r.precision,
        "recall": r.recall,
        "f1": r.f1,
        "roc_auc": r.roc_auc,
        "confusion_matrix": r.cm.tolist(),
    }
