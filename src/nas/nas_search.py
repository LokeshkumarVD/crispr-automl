# src/nas/nas_search.py
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.nas_model import NASModel
from src.nas.search_space import sample_architecture, mutate_architecture
from src.eval.metrics import regression_metrics

# ---- Reduced budget (a prototype, not the full-scale search) ----
N_TRAIN, N_VAL = 8000, 2000
EPOCHS, BATCH = 2, 64
N_RANDOM_INIT = 6   # random candidates first
N_TOTAL = 12        # total candidates (the rest are mutations of the best ones)
SEED = 42

LOG_PATH = Path("experiments/logs/nas_search_log.json")
FIG_PATH = Path("reports/figures/nas_convergence.png")

# The two baselines expressed in the same search space, so they can be
# evaluated under exactly the same mini budget as the NAS candidates.
REFERENCE_ARCHS = {
    "CNN-BiLSTM (baseline-equivalent)": [
        {"block_type": "cnn_small", "channels": 16, "dropout": 0.1},
        {"block_type": "bilstm", "channels": 64, "dropout": 0.0},
    ],
    "CNN-Transformer (baseline-equivalent)": [
        {"block_type": "cnn_small", "channels": 16, "dropout": 0.1},
        {"block_type": "attention", "channels": 16, "num_heads": 4, "dropout": 0.1},
    ],
}


def arch_to_str(arch):
    parts = []
    for l in arch:
        s = f"{l['block_type']}{l['channels']}"
        if l["block_type"] == "attention":
            s += f"h{l['num_heads']}"
        parts.append(s)
    return " > ".join(parts)


def load_data():
    d = np.load("data/processed/ontarget_splits.npz")

    def make(split, n):
        return TensorDataset(
            torch.tensor(d[f"X_seq_{split}"][:n], dtype=torch.float32),
            torch.tensor(d[f"X_aux_{split}"][:n], dtype=torch.float32),
            torch.tensor(d[f"y_{split}"][:n], dtype=torch.float32),
        )

    train = DataLoader(make("train", N_TRAIN), batch_size=BATCH, shuffle=True)
    val = DataLoader(make("val", N_VAL), batch_size=BATCH, shuffle=False)
    return train, val


def evaluate_architecture(arch, train_loader, val_loader):
    """Brief training, then validation Spearman = the fitness score."""
    torch.manual_seed(SEED)
    model = NASModel(arch)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for x_seq, x_aux, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x_seq, x_aux), y).backward()
            opt.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_seq, x_aux, y in val_loader:
            preds.extend(model(x_seq, x_aux).numpy())
            trues.extend(y.numpy())
    sp = regression_metrics(trues, preds)["spearman"]
    if np.isnan(sp):
        sp = -1.0
    return float(sp), n_params


def run_search():
    random.seed(SEED)
    np.random.seed(SEED)
    train_loader, val_loader = load_data()

    print("=== Reference baselines under the same mini budget ===")
    ref_scores = {}
    for name, arch in REFERENCE_ARCHS.items():
        score, n_params = evaluate_architecture(arch, train_loader, val_loader)
        ref_scores[name] = score
        print(f"{name}: fitness (val Spearman) = {score:.4f}, params = {n_params:,}")

    history = []

    def record(arch, origin):
        t0 = time.time()
        try:
            score, n_params = evaluate_architecture(arch, train_loader, val_loader)
        except Exception as e:
            print(f"  candidate failed: {e}")
            score, n_params = -1.0, 0
        history.append({"index": len(history) + 1, "origin": origin, "architecture": arch,
                        "fitness": score, "params": n_params})
        print(f"[{len(history)}/{N_TOTAL}] {origin:8s} fitness={score:.4f} params={n_params:,} "
              f"time={time.time()-t0:.0f}s :: {arch_to_str(arch)}")

    print("\n=== NAS search ===")
    for _ in range(N_RANDOM_INIT):
        record(sample_architecture(), "random")

    while len(history) < N_TOTAL:
        ranked = sorted(history, key=lambda h: h["fitness"], reverse=True)
        parent = random.choice(ranked[:2])["architecture"]
        child = mutate_architecture(parent)
        tries = 0
        while child == parent and tries < 10:
            child = mutate_architecture(parent, mutation_rate=0.5)
            tries += 1
        record(child, "mutated")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump({"history": history, "reference": ref_scores,
                   "budget": {"n_train": N_TRAIN, "n_val": N_VAL, "epochs": EPOCHS}}, f, indent=2)

    # Convergence plot
    idx = [h["index"] for h in history]
    fit = [h["fitness"] for h in history]
    best_so_far = np.maximum.accumulate(fit)
    plt.figure(figsize=(9, 5.5))
    for origin, marker in [("random", "o"), ("mutated", "s")]:
        xs = [h["index"] for h in history if h["origin"] == origin]
        ys = [h["fitness"] for h in history if h["origin"] == origin]
        plt.scatter(xs, ys, marker=marker, label=f"{origin} candidate", zorder=3)
    plt.plot(idx, best_so_far, linewidth=2, label="best so far", zorder=2)
    for name, s in ref_scores.items():
        plt.axhline(s, linestyle="--", alpha=0.7, label=f"{name}: {s:.3f}")
    plt.xlabel("Candidate #")
    plt.ylabel("Fitness (val Spearman, reduced budget)")
    plt.title("NAS convergence (prototype: 12 candidates, 2 epochs, 8k train samples)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()

    best = max(history, key=lambda h: h["fitness"])
    print(f"\nBEST: fitness={best['fitness']:.4f} :: {arch_to_str(best['architecture'])}")
    print(f"Log saved to {LOG_PATH}, plot saved to {FIG_PATH}")


if __name__ == "__main__":
    run_search()