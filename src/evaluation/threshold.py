from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

Metric = Literal["f1", "accuracy", "precision", "recall"]

@dataclass
class ThresholdRow:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float

def sweep_thresholds(y_true, y_prob, thresholds: np.ndarray) -> list[ThresholdRow]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    rows: list[ThresholdRow] = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            ThresholdRow(
                threshold=float(t),
                accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
            )
        )
    return rows

def pick_best_threshold(rows: list[ThresholdRow], metric: Metric = "f1") -> ThresholdRow:
    best = max(rows, key=lambda r: getattr(r, metric))
    return best

def rows_to_dicts(rows: list[ThresholdRow]):
    return [r.__dict__ for r in rows]
