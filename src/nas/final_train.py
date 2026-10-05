# src/nas/final_train.py
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.nas_model import NASModel
from src.eval.metrics import regression_metrics
from src.nas.nas_search import REFERENCE_ARCHS, arch_to_str, LOG_PATH

EPOCHS, BATCH, LR, SEED = 5, 32, 1e-3, 42
CKPT_PATH = Path("experiments/checkpoints/nas_best.pt")
RESULTS_PATH = Path("experiments/logs/final_comparison.json")


def make_loader(data, split, shuffle):
    ds = TensorDataset(
        torch.tensor(data[f"X_seq_{split}"], dtype=torch.float32),
        torch.tensor(data[f"X_aux_{split}"], dtype=torch.float32),
        torch.tensor(data[f"y_{split}"], dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_seq, x_aux, y in loader:
            preds.extend(model(x_seq, x_aux).numpy())
            trues.extend(y.numpy())
    return regression_metrics(trues, preds)


def train_full(name, arch, data):
    print(f"\n=== Training: {name} ===")
    print("Architecture:", arch_to_str(arch))
    torch.manual_seed(SEED)
    model = NASModel(arch)
    train_loader = make_loader(data, "train", True)
    val_loader = make_loader(data, "val", False)
    test_loader = make_loader(data, "test", False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x_seq, x_aux, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x_seq, x_aux), y)
            loss.backward()
            opt.step()
            total += loss.item()
        m = evaluate(model, val_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} - loss {total/len(train_loader):.4f} - val Spearman {m['spearman']:.4f}")

    return model, evaluate(model, val_loader), evaluate(model, test_loader)


if __name__ == "__main__":
    with open(LOG_PATH) as f:
        log = json.load(f)
    best = max(log["history"], key=lambda h: h["fitness"])
    data = dict(np.load("data/processed/ontarget_splits.npz"))

    runs = {
        "NAS-best": best["architecture"],
        "CNN-BiLSTM (baseline-equivalent)": REFERENCE_ARCHS["CNN-BiLSTM (baseline-equivalent)"],
    }
    results = {}
    for name, arch in runs.items():
        model, val_m, test_m = train_full(name, arch, data)
        results[name] = {
            "architecture": arch,
            "val_spearman": float(val_m["spearman"]), "val_pearson": float(val_m["pearson"]),
            "test_spearman": float(test_m["spearman"]), "test_pearson": float(test_m["pearson"]),
            "params": sum(p.numel() for p in model.parameters()),
        }
        if name == "NAS-best":
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"architecture": arch, "state_dict": model.state_dict()}, CKPT_PATH)

    print("\n=== FINAL COMPARISON (same training settings) ===")
    print(f"{'Model':40s} {'Val Spearman':>13s} {'Test Spearman':>14s} {'Test Pearson':>13s} {'Params':>9s}")
    for name, r in results.items():
        print(f"{name:40s} {r['val_spearman']:13.4f} {r['test_spearman']:14.4f} "
              f"{r['test_pearson']:13.4f} {r['params']:9,d}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH} and model to {CKPT_PATH}")