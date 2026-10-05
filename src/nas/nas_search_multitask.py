# src/nas/nas_search_multitask.py
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.nas_model_multitask import MultiTaskNASModel
from src.models.train_multitask import train_multitask
from src.nas.search_space import sample_architecture, mutate_architecture
from src.nas.nas_search import arch_to_str

N_RANDOM_INIT = 4
N_TOTAL = 8          # kept small — each candidate now trains on BOTH tasks, so this is slower
EPOCHS_PER_CANDIDATE = 2
ALPHA = 0.7
SEED = 42

LOG_PATH = Path("experiments/logs/nas_multitask_search_log.json")
FIG_PATH = Path("reports/figures/nas_multitask_convergence.png")


def fitness_fn(ot_spearman, off_auprc):
    """Combined fitness: average of on-target Spearman and off-target AUPRC,
    so a candidate must do reasonably well on BOTH tasks to score well —
    this is what actually makes it a multi-task search, not just on-target again."""
    ot_spearman = max(ot_spearman, 0.0)  # floor negative correlation at 0 for scoring
    return (ot_spearman + off_auprc) / 2


def evaluate_candidate(arch):
    torch.manual_seed(SEED)
    model = MultiTaskNASModel(arch)
    try:
        ot_metrics, off_metrics = train_multitask(
            model, epochs=EPOCHS_PER_CANDIDATE, alpha=ALPHA
        )
        fitness = fitness_fn(ot_metrics["spearman"], off_metrics["auprc"])
        n_params = sum(p.numel() for p in model.parameters())
        return fitness, ot_metrics["spearman"], off_metrics["auprc"], n_params
    except Exception as e:
        print(f"  candidate failed: {e}")
        return -1.0, -1.0, 0.0, 0


def run_search():
    random.seed(SEED)
    np.random.seed(SEED)
    history = []

    def record(arch, origin):
        t0 = time.time()
        fitness, ot_sp, off_ap, n_params = evaluate_candidate(arch)
        history.append({
            "index": len(history) + 1, "origin": origin, "architecture": arch,
            "fitness": fitness, "ontarget_spearman": ot_sp, "offtarget_auprc": off_ap,
            "params": n_params,
        })
        print(f"[{len(history)}/{N_TOTAL}] {origin:8s} fitness={fitness:.4f} "
              f"(OnT={ot_sp:.4f}, OffT={off_ap:.4f}) params={n_params:,} "
              f"time={time.time()-t0:.0f}s :: {arch_to_str(arch)}")

    print("=== Multi-task NAS search ===")
    for _ in range(N_RANDOM_INIT):
        record(sample_architecture(), "random")

    while len(history) < N_TOTAL:
        ranked = sorted(history, key=lambda h: h["fitness"], reverse=True)
        parent = random.choice(ranked[:2])["architecture"]
        child = mutate_architecture(parent)
        record(child, "mutated")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump({"history": history, "alpha": ALPHA}, f, indent=2)

    idx = [h["index"] for h in history]
    fit = [h["fitness"] for h in history]
    best_so_far = np.maximum.accumulate(fit)
    plt.figure(figsize=(9, 5.5))
    for origin, marker in [("random", "o"), ("mutated", "s")]:
        xs = [h["index"] for h in history if h["origin"] == origin]
        ys = [h["fitness"] for h in history if h["origin"] == origin]
        plt.scatter(xs, ys, marker=marker, label=f"{origin} candidate", zorder=3)
    plt.plot(idx, best_so_far, linewidth=2, label="best so far", zorder=2)
    plt.xlabel("Candidate #")
    plt.ylabel("Fitness = avg(on-target Spearman, off-target AUPRC)")
    plt.title("Multi-task NAS convergence (prototype)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()

    best = max(history, key=lambda h: h["fitness"])
    print(f"\nBEST: fitness={best['fitness']:.4f} (OnT={best['ontarget_spearman']:.4f}, "
          f"OffT={best['offtarget_auprc']:.4f}) :: {arch_to_str(best['architecture'])}")
    print(f"Log saved to {LOG_PATH}, plot saved to {FIG_PATH}")


if __name__ == "__main__":
    run_search()