# src/nas/rerank_top_candidates.py
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.models.nas_model_multitask import MultiTaskNASModel
from src.models.train_multitask import train_multitask
from src.nas.nas_search import arch_to_str

SEARCH_LOG_PATH = Path("experiments/logs/nas_multitask_search_log_val.json")
OUT_PATH = Path("experiments/logs/rerank_top_candidates.json")

TOP_K = 3                      # top candidates by the search's validation fitness
RERANK_SEEDS = [11, 22, 33]    # fresh seeds (not the search seed 456, not 123/7/2024)
EPOCHS = 5
ALPHA = 0.7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fitness_fn(val_spearman, val_auprc):
    """Same formula as the search: avg(on-target Spearman, off-target AUPRC),
    both on VALIDATION data. NaN or negative correlations are floored at 0."""
    if val_spearman is None or np.isnan(val_spearman):
        val_spearman = 0.0
    if val_auprc is None or np.isnan(val_auprc):
        val_auprc = 0.0
    return (max(val_spearman, 0.0) + val_auprc) / 2


def mean_std(values):
    values = np.array(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return float(values.mean()), float(std)


if __name__ == "__main__":
    with open(SEARCH_LOG_PATH) as f:
        log = json.load(f)

    top = sorted(log["history"], key=lambda h: h["fitness"], reverse=True)[:TOP_K]

    print("Re-ranking the top candidates from the validation search")
    print("(validation metrics only; the test set is never used here)\n")
    for h in top:
        print(f"  search #{h['index']}: {arch_to_str(h['architecture'])} "
              f"(single-seed validation fitness {h['fitness']:.4f})")

    results = []
    for h in top:
        arch = h["architecture"]
        runs = []
        for seed in RERANK_SEEDS:
            print(f"\n--- search #{h['index']} | seed {seed} | {arch_to_str(arch)} ---")
            set_seed(seed)
            model = MultiTaskNASModel(arch)
            r = train_multitask(model, epochs=EPOCHS, alpha=ALPHA, verbose=False)
            vs = float(r["val_ot"]["spearman"])
            va = float(r["val_off"]["auprc"])
            fit = fitness_fn(vs, va)
            runs.append({"seed": seed, "val_spearman": vs, "val_auprc": va, "fitness": fit})
            print(f"val Spearman {vs:.4f} | val AUPRC {va:.4f} | fitness {fit:.4f}")

        fit_m, fit_s = mean_std([r["fitness"] for r in runs])
        sp_m, sp_s = mean_std([r["val_spearman"] for r in runs])
        ap_m, ap_s = mean_std([r["val_auprc"] for r in runs])
        results.append({
            "search_index": h["index"],
            "architecture": arch,
            "params": h["params"],
            "single_seed_search_fitness": h["fitness"],
            "runs": runs,
            "mean_fitness": fit_m, "std_fitness": fit_s,
            "mean_val_spearman": sp_m, "std_val_spearman": sp_s,
            "mean_val_auprc": ap_m, "std_val_auprc": ap_s,
        })

    results.sort(key=lambda r: r["mean_fitness"], reverse=True)
    selected = results[0]

    print("\n" + "=" * 86)
    print(f"RE-RANKING (validation only, seeds {RERANK_SEEDS})")
    print("=" * 86)
    print(f"{'Architecture':32s} {'Search fit':>11s} {'Val Spearman':>18s} {'Val AUPRC':>16s} {'Mean fit':>9s}")
    for r in results:
        print(f"{arch_to_str(r['architecture']):32s} {r['single_seed_search_fitness']:11.4f} "
              f"{r['mean_val_spearman']:11.4f} +/- {r['std_val_spearman']:.4f} "
              f"{r['mean_val_auprc']:9.4f} +/- {r['std_val_auprc']:.4f} {r['mean_fitness']:9.4f}")

    print()
    print(f"SELECTED: {arch_to_str(selected['architecture'])}")
    print(f"  mean validation fitness {selected['mean_fitness']:.4f} +/- {selected['std_fitness']:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "protocol": "top-k re-evaluation on fresh seeds, validation only (test set not used)",
            "seeds": RERANK_SEEDS,
            "epochs": EPOCHS,
            "alpha": ALPHA,
            "selected": selected,
            "ranking": results,
        }, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")