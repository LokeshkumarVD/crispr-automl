# src/models/evaluate_test_set.py
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.nas_model import NASModel
from src.eval.metrics import regression_metrics

CKPT_PATH = "experiments/checkpoints/nas_best.pt"
FIG_PATH = Path("reports/figures/test_set_scatter.png")


def main():
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model = NASModel(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = np.load("data/processed/ontarget_splits.npz")
    x_seq = torch.tensor(data["X_seq_test"], dtype=torch.float32)
    x_aux = torch.tensor(data["X_aux_test"], dtype=torch.float32)
    y_true = data["y_test"]

    with torch.no_grad():
        y_pred = model(x_seq, x_aux).numpy()

    metrics = regression_metrics(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Test set size: {len(y_true)}")
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"Pearson: {metrics['pearson']:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.15, s=8)
    plt.plot([0, 1], [0, 1], "r--", label="perfect prediction")
    plt.xlabel("True efficiency")
    plt.ylabel("Predicted efficiency")
    plt.title(f"NAS-best on held-out test set (n={len(y_true)})\nSpearman={metrics['spearman']:.3f}, MAE={mae:.3f}")
    plt.legend()
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved scatter plot to {FIG_PATH}")


if __name__ == "__main__":
    main()