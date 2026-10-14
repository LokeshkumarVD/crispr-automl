# src/models/multiseed_comparison.py
import json
from pathlib import Path

import numpy as np

import src.models.final_baseline_comparison as fbc

NEW_SEEDS = [7, 2024]  # independent of the search seed (456) and of seed 123
RESULTS_PATH = Path("experiments/logs/multiseed_comparison.json")

MODELS = {
    "Original CNN": fbc.CNN_BASELINE,
    "CNN + BiLSTM": fbc.BILSTM_BASELINE,
    "NAS winner (from log)": fbc.NAS_WINNER,
}

# Seed 123 results copied from the earlier terminal output of
# final_baseline_comparison.py (not recomputed here).
SEED_123 = {
    "Original CNN": {"ot": 0.3106, "off": 0.8351},
    "CNN + BiLSTM": {"ot": 0.1916, "off": 0.8164},
    "NAS winner (from log)": {"ot": 0.7640, "off": 0.9537},
}


def mean_std(values):
    values = np.array(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return values.mean(), std


if __name__ == "__main__":
    per_seed = {name: {123: SEED_123[name]} for name in MODELS}

    for seed in NEW_SEEDS:
        fbc.SEED = seed  # run_experiment() reads this module-level value
        for name, arch in MODELS.items():
            print(f"\n##### seed {seed} :: {name} #####")
            r = fbc.run_experiment(name, arch)
            per_seed[name][seed] = {
                "ot": float(r["ot_spearman"]),
                "off": float(r["off_auprc"]),
            }

    print("\n" + "=" * 78)
    print("MULTI-SEED COMPARISON (test set, 3 seeds: 123, 7, 2024)")
    print("=" * 78)
    print(f"{'Model':26s} {'Seed':>6s} {'OnT Spearman':>14s} {'OffT AUPRC':>12s}")
    for name in MODELS:
        for seed, r in per_seed[name].items():
            print(f"{name:26s} {seed:6d} {r['ot']:14.4f} {r['off']:12.4f}")

    print()
    print(f"{'Model':26s} {'OnT Spearman (mean +/- std)':>30s} {'OffT AUPRC (mean +/- std)':>28s}")
    summary = {}
    for name in MODELS:
        ot_m, ot_s = mean_std([r["ot"] for r in per_seed[name].values()])
        off_m, off_s = mean_std([r["off"] for r in per_seed[name].values()])
        summary[name] = {
            "ot_mean": ot_m, "ot_std": ot_s, "off_mean": off_m, "off_std": off_s,
            "per_seed": {str(k): v for k, v in per_seed[name].items()},
        }
        print(f"{name:26s} {ot_m:14.4f} +/- {ot_s:.4f} {off_m:14.4f} +/- {off_s:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")