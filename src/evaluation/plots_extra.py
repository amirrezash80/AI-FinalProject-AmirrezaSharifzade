from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import calibration_curve

def save_f1_vs_threshold(thresholds: np.ndarray, f1s: np.ndarray, out_path: str | Path, title: str = "F1 vs Threshold"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(thresholds, f1s)
    ax.set_xlabel("threshold")
    ax.set_ylabel("F1")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_precision_recall_curve(y_true, y_prob, out_path: str | Path, title: str = "Precision-Recall Curve"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_calibration_curve(y_true, y_prob, out_path: str | Path, title: str = "Calibration Curve", n_bins: int = 10):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
