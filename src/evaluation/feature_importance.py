from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def topk_importances(importances: np.ndarray, feature_names: list[str], k: int = 20):
    idx = np.argsort(np.abs(importances))[::-1][:k]
    return [(feature_names[i], float(importances[i])) for i in idx]

def save_barh(items: list[tuple[str, float]], out_path: str | Path, title: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [x[0] for x in items][::-1]
    vals = [x[1] for x in items][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(labels))))
    ax.barh(labels, vals)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
