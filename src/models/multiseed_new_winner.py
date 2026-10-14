# src/models/multiseed_new_winner.py
import json
from pathlib import Path

import numpy as np

import src.models.final_baseline_comparison as fbc

NEW_LOG_PATH = Path("experiments/logs/nas_multitask_search_log_val.json")
PRIOR_RESULTS_PATH = Path("experiments/logs/multiseed_comparison.json")
RESULTS_PATH = Path("experiments/logs/multiseed_new_winner.json")

# Same fresh seeds used for the other models. The search seed (456) is excluded,
# because the new winner was selected and tested once at that seed.
SEEDS = [123, 7, 2024]

NEW_NAME = "NAS winner v2 (validation-selected)"


def load_new_winner():
    with open(NEW_LOG_PATH) as f:
        log = json.load(f)
    if "winner" in log:
        return log["winner"]["architecture"]
    return max(log["history"], key=lambda h: h["fitness"])["architecture"]


def load_prior_results():
    """Per-seed numbers saved by multiseed_comparison.py (baselines + old winner)."""
    try:
        with open(PRIOR_RESULTS_PATH) as f:
            summary = json.load(f)
    except FileNotFoundError:
        print(f"Note: {PRIOR_RESULTS_PATH} not found; showing the new winner only.")
        return {}
    prior = {}
    for name, entry in summary.items():
        prior[name] = {int(seed): r for seed, r in entry["per_seed"].items()}
    return prior


def mean_std(values):
    values = np.array(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return values.mean(), std


if __name__ == "__main__":
    new_arch = load_new_winner()
    print("New winner loaded from:", NEW_LOG_PATH)
    print("Architecture:")
    for i, layer in enumerate(new_arch, start=1):
        print(f"  Layer {i}: {layer}")

    prior = load_prior_results()

    new_per_seed = {}
    for seed in SEEDS:
        fbc.SEED = seed  # run_experiment() reads this module-level value
        print(f"\n##### seed {seed} :: {NEW_NAME} #####")
        r = fbc.run_experiment(NEW_NAME, new_arch)
        new_per_seed[seed] = {
            "ot": float(r["ot_spearman"]),
            "off": float(r["off_auprc"]),
        }

    # Order for the final table. "NAS winner (from log)" is the earlier, test-selected winner.
    all_results = {}
    for name in ["Original CNN", "CNN + BiLSTM", "NAS winner (from log)"]:
        if name in prior:
            label = "NAS winner v1 (test-selected)" if name.startswith("NAS winner") else name
            all_results[label] = {s: prior[name][s] for s in SEEDS if s in prior[name]}
    all_results[NEW_NAME] = new_per_seed

    print("\n" + "=" * 80)
    print("MULTI-SEED COMPARISON (test set, seeds 123 / 7 / 2024)")
    print("=" * 80)
    print(f"{'Model':36s} {'Seed':>6s} {'OnT Spearman':>14s} {'OffT AUPRC':>12s}")
    for name, per_seed in all_results.items():
        for seed, r in per_seed.items():
            print(f"{name:36s} {seed:6d} {r['ot']:14.4f} {r['off']:12.4f}")

    print()
    print(f"{'Model':36s} {'OnT Spearman (mean +/- std)':>30s} {'OffT AUPRC (mean +/- std)':>28s}")
    summary = {}
    for name, per_seed in all_results.items():
        ot_m, ot_s = mean_std([r["ot"] for r in per_seed.values()])
        off_m, off_s = mean_std([r["off"] for r in per_seed.values()])
        summary[name] = {
            "ot_mean": ot_m, "ot_std": ot_s, "off_mean": off_m, "off_std": off_s,
            "per_seed": {str(k): v for k, v in per_seed.items()},
        }
        print(f"{name:36s} {ot_m:14.4f} +/- {ot_s:.4f} {off_m:14.4f} +/- {off_s:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")