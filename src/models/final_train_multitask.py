# src/models/final_train_multitask.py
import itertools
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics
from src.models.nas_model_multitask import MultiTaskNASModel
from src.models.train_multitask import ONTARGET_NPZ, get_offtarget_loaders, evaluate_split
from src.nas.nas_search import arch_to_str


# ============================================================
# SETTINGS
# ============================================================

SELECTION_PATH = Path("experiments/logs/rerank_top_candidates.json")
CKPT_DIR = Path("experiments/checkpoints")
LOG_PATH = Path("experiments/logs/final_training_results.json")
FIG_DIR = Path("reports/figures")

# Seeds not used earlier in this project. One seed is the fast path (about 15-30 min).
# For mean +/- std, use [2025, 2026, 2027] (several times longer).
SEEDS = [2025]

MAX_EPOCHS = 30
PATIENCE = 5          # stop if validation fitness has not improved for this many epochs
MIN_DELTA = 1e-4
BATCH_SIZE = 32
LR = 1e-3
ALPHA = 0.7           # same joint-loss weighting as the search


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fitness_fn(val_spearman, val_auprc):
    """Same formula as the search and the re-ranking (validation data only)."""
    if val_spearman is None or np.isnan(val_spearman):
        val_spearman = 0.0
    if val_auprc is None or np.isnan(val_auprc):
        val_auprc = 0.0
    return (max(val_spearman, 0.0) + val_auprc) / 2


def load_selected_architecture():
    with open(SELECTION_PATH) as f:
        return json.load(f)["selected"]["architecture"]


# ============================================================
# TRAINING WITH EARLY STOPPING (validation only)
# ============================================================

def train_one_seed(arch, seed):
    set_seed(seed)
    model = MultiTaskNASModel(arch)

    ot_train = get_dataloader(ONTARGET_NPZ, "train", BATCH_SIZE, shuffle=True)
    off_train, _, _, pos_weight = get_offtarget_loaders(BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Same loss normalization as in the search
    running_ot, running_off, momentum = 1.0, 1.0, 0.9

    history = []
    best = {"fitness": -1.0, "epoch": 0, "state": None, "val_ot": None, "val_off": None}
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_ot, total_off, n_steps = 0.0, 0.0, 0

        off_cycle = itertools.cycle(off_train)
        for (x_seq, x_aux, y_ot), (x_pair, y_off) in zip(ot_train, off_cycle):
            optimizer.zero_grad()

            pred_ot = model.forward_ontarget(x_seq, x_aux)
            pred_off = model.forward_offtarget(x_pair)

            loss_ot = mse_loss(pred_ot, y_ot)
            loss_off = bce_loss(pred_off, y_off)

            running_ot = momentum * running_ot + (1 - momentum) * loss_ot.item()
            running_off = momentum * running_off + (1 - momentum) * loss_off.item()

            loss = (ALPHA * loss_ot / (running_ot + 1e-8)
                    + (1 - ALPHA) * loss_off / (running_off + 1e-8))
            loss.backward()
            optimizer.step()

            total_ot += loss_ot.item()
            total_off += loss_off.item()
            n_steps += 1

        val_ot, val_off = evaluate_split(model, "val", BATCH_SIZE)
        fit = fitness_fn(val_ot["spearman"], val_off["auprc"])

        history.append({
            "epoch": epoch,
            "train_ot_loss": total_ot / n_steps,
            "train_off_loss": total_off / n_steps,
            "val_spearman": float(val_ot["spearman"]),
            "val_pearson": float(val_ot["pearson"]),
            "val_auprc": float(val_off["auprc"]),
            "val_fitness": float(fit),
        })
        print(f"  Epoch {epoch:2d}/{MAX_EPOCHS} | OT loss {total_ot / n_steps:.4f} | "
              f"OFF loss {total_off / n_steps:.4f} | val Spearman {val_ot['spearman']:.4f} | "
              f"val AUPRC {val_off['auprc']:.4f} | val fitness {fit:.4f} | {time.time() - t0:.0f}s")

        if fit > best["fitness"] + MIN_DELTA:
            best.update({
                "fitness": float(fit),
                "epoch": epoch,
                "state": {k: v.clone() for k, v in model.state_dict().items()},
                "val_ot": {k: float(v) for k, v in val_ot.items()},
                "val_off": {k: float(v) for k, v in val_off.items()},
            })
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Early stopping: no improvement for {PATIENCE} epochs.")
                break

    model.load_state_dict(best["state"])  # restore the best epoch
    print(f"  Restored best epoch: {best['epoch']} (val fitness {best['fitness']:.4f})")
    return model, history, best


# ============================================================
# TEST EVALUATION (called once per trained model, after training)
# ============================================================

@torch.no_grad()
def predict_test(model):
    model.eval()
    ot_loader = get_dataloader(ONTARGET_NPZ, "test", BATCH_SIZE, shuffle=False)
    _, _, off_test, _ = get_offtarget_loaders(BATCH_SIZE)

    ot_true, ot_pred, off_true, off_prob = [], [], [], []
    for x_seq, x_aux, y in ot_loader:
        ot_pred.extend(model.forward_ontarget(x_seq, x_aux).numpy())
        ot_true.extend(y.numpy())
    for x_pair, y in off_test:
        off_prob.extend(torch.sigmoid(model.forward_offtarget(x_pair)).numpy())
        off_true.extend(y.numpy())

    return (np.array(ot_true), np.array(ot_pred), np.array(off_true), np.array(off_prob))


def compute_test_metrics(preds):
    ot_true, ot_pred, off_true, off_prob = preds
    reg = regression_metrics(ot_true, ot_pred)
    cls = classification_metrics(off_true, off_prob)
    return {
        "ot_spearman": float(reg["spearman"]),
        "ot_pearson": float(reg["pearson"]),
        "ot_mae": float(np.mean(np.abs(ot_true - ot_pred))),
        "ot_rmse": float(np.sqrt(np.mean((ot_true - ot_pred) ** 2))),
        "off_auprc": float(cls["auprc"]),
    }


# ============================================================
# PLOTS
# ============================================================

def plot_curves(history, best_epoch, arch_label, seed, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(epochs, [h["train_ot_loss"] for h in history], label="on-target (MSE)")
    axes[0].plot(epochs, [h["train_off_loss"] for h in history], label="off-target (weighted BCE)")
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, [h["val_spearman"] for h in history], color="tab:green")
    axes[1].set_title("Validation on-target Spearman")

    axes[2].plot(epochs, [h["val_auprc"] for h in history], color="tab:red")
    axes[2].set_title("Validation off-target AUPRC")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.axvline(best_epoch, linestyle="--", alpha=0.6)
    fig.suptitle(f"Final training: {arch_label} (seed {seed}); dashed line = selected epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test(preds, metrics, arch_label, path):
    ot_true, ot_pred, off_true, off_prob = preds
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].scatter(ot_true, ot_pred, s=6, alpha=0.15)
    axes[0].plot([0, 1], [0, 1], "r--", label="perfect prediction")
    axes[0].set_xlabel("True efficiency")
    axes[0].set_ylabel("Predicted efficiency")
    axes[0].set_title(f"On-target test set (n={len(ot_true)})\n"
                      f"Spearman {metrics['ot_spearman']:.3f}, MAE {metrics['ot_mae']:.3f}")
    axes[0].legend(fontsize=8)

    precision, recall, _ = precision_recall_curve(off_true, off_prob)
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Off-target test set ({int(off_true.sum())} positives of {len(off_true)})\n"
                      f"AUPRC {metrics['off_auprc']:.3f}")

    fig.suptitle(f"Final model: {arch_label}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    arch = load_selected_architecture()
    arch_label = arch_to_str(arch)

    print("Final architecture (selected on validation, from the re-ranking):")
    for i, layer in enumerate(arch, start=1):
        print(f"  Layer {i}: {layer}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = {}
    for seed in SEEDS:
        print(f"\n===== Training seed {seed} =====")
        model, history, best = train_one_seed(arch, seed)

        ckpt_path = CKPT_DIR / f"final_multitask_seed{seed}.pt"
        torch.save({
            "architecture": arch,
            "state_dict": model.state_dict(),
            "seed": seed,
            "best_epoch": best["epoch"],
            "val_fitness": best["fitness"],
        }, ckpt_path)
        plot_curves(history, best["epoch"], arch_label, seed,
                    FIG_DIR / f"final_training_curves_seed{seed}.png")
        runs[seed] = {"model": model, "history": history, "best": best, "ckpt": ckpt_path}

    # The "final" model is the seed with the best VALIDATION fitness (never test)
    final_seed = max(runs, key=lambda s: runs[s]["best"]["fitness"])
    shutil.copyfile(runs[final_seed]["ckpt"], CKPT_DIR / "final_multitask.pt")

    # Test set: each trained model is evaluated once, after all training is finished
    print("\nEvaluating on the test set (once per trained model)...")
    results = {}
    final_preds = None
    for seed, run in runs.items():
        preds = predict_test(run["model"])
        results[seed] = compute_test_metrics(preds)
        if seed == final_seed:
            final_preds = preds

    plot_test(final_preds, results[final_seed], arch_label, FIG_DIR / "final_test_plots.png")

    print("\n" + "=" * 92)
    print(f"FINAL TEST RESULTS: {arch_label}")
    print("=" * 92)
    print(f"{'Seed':>6s} {'Best ep':>8s} {'Val fit':>8s} {'OnT Spearman':>13s} "
          f"{'OnT Pearson':>12s} {'MAE':>7s} {'RMSE':>7s} {'OffT AUPRC':>11s}")
    for seed, r in results.items():
        b = runs[seed]["best"]
        mark = "  <- final model" if seed == final_seed else ""
        print(f"{seed:6d} {b['epoch']:8d} {b['fitness']:8.4f} {r['ot_spearman']:13.4f} "
              f"{r['ot_pearson']:12.4f} {r['ot_mae']:7.4f} {r['ot_rmse']:7.4f} {r['off_auprc']:11.4f}{mark}")

    if len(results) > 1:
        for key, label in [("ot_spearman", "On-target Spearman"), ("off_auprc", "Off-target AUPRC")]:
            vals = np.array([r[key] for r in results.values()])
            print(f"{label}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f} over {len(vals)} seeds")

    with open(LOG_PATH, "w") as f:
        json.dump({
            "architecture": arch,
            "protocol": "early stopping on validation fitness; test evaluated once per trained model",
            "settings": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
                         "lr": LR, "alpha": ALPHA, "seeds": SEEDS},
            "final_seed": final_seed,
            "runs": {str(s): {
                "best_epoch": runs[s]["best"]["epoch"],
                "val_fitness": runs[s]["best"]["fitness"],
                "val_ontarget": runs[s]["best"]["val_ot"],
                "val_offtarget": runs[s]["best"]["val_off"],
                "test": results[s],
                "history": runs[s]["history"],
            } for s in runs},
        }, f, indent=2)

    print(f"\nFinal checkpoint: {CKPT_DIR / 'final_multitask.pt'}")
    print(f"Curves:           {FIG_DIR / 'final_training_curves_seed*.png'}")
    print(f"Test plots:       {FIG_DIR / 'final_test_plots.png'}")
    print(f"Results log:      {LOG_PATH}")