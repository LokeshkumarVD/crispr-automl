# src/nas/final_multiseed.py
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.nas_model import NASModel
from src.nas.nas_search import REFERENCE_ARCHS, arch_to_str, LOG_PATH
from src.nas.final_train import make_loader, evaluate, EPOCHS, LR

SEEDS = [0, 1, 2]
RESULTS_PATH = Path("experiments/logs/final_multiseed.json")


def train_once(arch, data, seed):
    """Train one model from scratch with a given seed, return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NASModel(arch)
    train_loader = make_loader(data, "train", True)
    test_loader = make_loader(data, "test", False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        for x_seq, x_aux, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x_seq, x_aux), y).backward()
            opt.step()
    return evaluate(model, test_loader)


if __name__ == "__main__":
    with open(LOG_PATH) as f:
        log = json.load(f)
    best = max(log["history"], key=lambda h: h["fitness"])
    data = dict(np.load("data/processed/ontarget_splits.npz"))

    models = {
        "NAS-best": best["architecture"],
        "CNN-BiLSTM (baseline-equivalent)": REFERENCE_ARCHS["CNN-BiLSTM (baseline-equivalent)"],
        "CNN-Transformer (baseline-equivalent)": REFERENCE_ARCHS["CNN-Transformer (baseline-equivalent)"],
    }

    results = {}
    for name, arch in models.items():
        print(f"\n=== {name} ===")
        print("Architecture:", arch_to_str(arch))
        spearmans, pearsons = [], []
        for seed in SEEDS:
            m = train_once(arch, data, seed)
            spearmans.append(float(m["spearman"]))
            pearsons.append(float(m["pearson"]))
            print(f"  seed {seed}: test Spearman {m['spearman']:.4f}, Pearson {m['pearson']:.4f}")
        results[name] = {
            "architecture": arch,
            "test_spearman_runs": spearmans,
            "test_pearson_runs": pearsons,
            "spearman_mean": float(np.mean(spearmans)),
            "spearman_std": float(np.std(spearmans, ddof=1)),
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_std": float(np.std(pearsons, ddof=1)),
        }

    print("\n=== FINAL MULTI-SEED COMPARISON (test set, mean +/- std over seeds) ===")
    print(f"{'Model':40s} {'Test Spearman':>20s} {'Test Pearson':>20s}")
    for name, r in results.items():
        print(f"{name:40s} {r['spearman_mean']:.4f} +/- {r['spearman_std']:.4f}   "
              f"{r['pearson_mean']:.4f} +/- {r['pearson_std']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")