# src/nas/nas_search_multitask.py
import copy
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
from src.models.train_multitask import train_multitask, evaluate_split
from src.nas.search_space import sample_architecture, mutate_architecture
from src.nas.nas_search import arch_to_str


# ============================================================
# SEARCH SETTINGS
# ============================================================

N_RANDOM_INIT = 4
N_TOTAL = 8
EPOCHS_PER_CANDIDATE = 5
ALPHA = 0.7

# Same seed for every candidate so architectures are compared under
# the same initialization / data-order randomness.
SEED = 456

# New file names, so the earlier (test-selected) log is NOT overwritten.
LOG_PATH = Path("experiments/logs/nas_multitask_search_log_val.json")
FIG_PATH = Path("reports/figures/nas_multitask_convergence_val.png")


# ============================================================
# FITNESS (VALIDATION ONLY)
# ============================================================

def fitness_fn(val_spearman, val_auprc):
    """avg(on-target Spearman, off-target AUPRC), both measured on VALIDATION data.
    Negative / NaN correlations are floored at 0 for scoring."""
    if val_spearman is None or np.isnan(val_spearman):
        val_spearman = 0.0
    if val_auprc is None or np.isnan(val_auprc):
        val_auprc = 0.0
    return (max(val_spearman, 0.0) + val_auprc) / 2


def arch_key(arch):
    """Canonical string for an architecture, used to detect duplicates."""
    return json.dumps(arch, sort_keys=True)


# ============================================================
# CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(arch):
    """Train one architecture and score it on VALIDATION data only.

    RNGs are reset to SEED for the candidate's own training, then restored,
    so candidate training does not change which architectures the search
    samples or mutates next."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        model = MultiTaskNASModel(arch)
        result = train_multitask(model, epochs=EPOCHS_PER_CANDIDATE, alpha=ALPHA)

        val_ot, val_off = result["val_ot"], result["val_off"]
        return {
            "fitness": fitness_fn(val_ot["spearman"], val_off["auprc"]),
            "val_spearman": float(val_ot["spearman"]),
            "val_auprc": float(val_off["auprc"]),
            "params": sum(p.numel() for p in model.parameters()),
            "state": copy.deepcopy(model.state_dict()),
        }

    except Exception as e:
        print(f"  candidate failed: {e}")
        return {"fitness": -1.0, "val_spearman": -1.0, "val_auprc": 0.0,
                "params": 0, "state": None}

    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


# ============================================================
# PROPOSING NEW CANDIDATES (never repeats an architecture)
# ============================================================

def propose_random(seen):
    for _ in range(200):
        arch = sample_architecture()
        if arch_key(arch) not in seen:
            return arch
    raise RuntimeError("Could not find an unseen random architecture")


def propose_mutation(history, seen):
    ranked = sorted(history, key=lambda h: h["fitness"], reverse=True)
    parent = copy.deepcopy(random.choice(ranked[:2])["architecture"])

    for attempt in range(30):
        rate = 0.3 if attempt < 10 else 0.6
        child = mutate_architecture(parent, mutation_rate=rate)
        if arch_key(child) not in seen:
            return child, "mutated"

    # Fallback if mutation keeps landing on already-tried architectures
    return propose_random(seen), "random"


# ============================================================
# NAS SEARCH
# ============================================================

def run_search():
    random.seed(SEED)
    np.random.seed(SEED)

    history = []
    seen = set()
    best = {"fitness": -float("inf"), "architecture": None, "state": None}

    def save_progress(winner=None):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "selection": "validation (test set not used for selection)",
            "alpha": ALPHA,
            "seed": SEED,
            "epochs_per_candidate": EPOCHS_PER_CANDIDATE,
            "history": history,
        }
        if winner is not None:
            payload["winner"] = winner
        with open(LOG_PATH, "w") as f:
            json.dump(payload, f, indent=2)

    def record(arch, origin):
        t0 = time.time()
        r = evaluate_candidate(arch)
        seen.add(arch_key(arch))

        history.append({
            "index": len(history) + 1,
            "origin": origin,
            "architecture": arch,
            "fitness": r["fitness"],
            "val_ontarget_spearman": r["val_spearman"],
            "val_offtarget_auprc": r["val_auprc"],
            "params": r["params"],
        })

        if r["state"] is not None and r["fitness"] > best["fitness"]:
            best["fitness"] = r["fitness"]
            best["architecture"] = arch
            best["state"] = r["state"]

        print(f"[{len(history)}/{N_TOTAL}] {origin:8s} fitness={r['fitness']:.4f} "
              f"(val OnT={r['val_spearman']:.4f}, val OffT={r['val_auprc']:.4f}) "
              f"params={r['params']:,} time={time.time() - t0:.0f}s :: {arch_to_str(arch)}")
        save_progress()  # write after every candidate so a long run is never lost

    print("=== Multi-task NAS search (selection on VALIDATION metrics) ===")

    for _ in range(N_RANDOM_INIT):
        record(propose_random(seen), "random")

    while len(history) < N_TOTAL:
        child, origin = propose_mutation(history, seen)
        record(child, origin)

    if best["state"] is None:
        raise RuntimeError("Every candidate failed; nothing to report.")

    # --------------------------------------------------------
    # Evaluate the chosen winner on the TEST set exactly once
    # --------------------------------------------------------
    winner_model = MultiTaskNASModel(best["architecture"])
    winner_model.load_state_dict(best["state"])
    test_ot, test_off = evaluate_split(winner_model, "test")

    winner = {
        "architecture": best["architecture"],
        "validation_fitness": best["fitness"],
        "test_ontarget_spearman": float(test_ot["spearman"]),
        "test_ontarget_pearson": float(test_ot["pearson"]),
        "test_offtarget_auprc": float(test_off["auprc"]),
    }
    save_progress(winner=winner)

    # --------------------------------------------------------
    # Convergence plot (validation fitness)
    # --------------------------------------------------------
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx = [h["index"] for h in history]
    fit = [h["fitness"] for h in history]
    best_so_far = np.maximum.accumulate(fit)

    plt.figure(figsize=(9, 5.5))
    for origin, marker in [("random", "o"), ("mutated", "s")]:
        xs = [h["index"] for h in history if h["origin"] == origin]
        ys = [h["fitness"] for h in history if h["origin"] == origin]
        if xs:
            plt.scatter(xs, ys, marker=marker, label=f"{origin} candidate", zorder=3)
    plt.plot(idx, best_so_far, linewidth=2, label="best so far", zorder=2)
    plt.xlabel("Candidate #")
    plt.ylabel("Validation fitness = avg(on-target Spearman, off-target AUPRC)")
    plt.title("Multi-task NAS convergence (validation-selected)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("BEST ARCHITECTURE (selected on validation)")
    print("=" * 60)
    print(f"Architecture:        {arch_to_str(best['architecture'])}")
    print(f"Validation fitness:  {best['fitness']:.4f}")
    print(f"Parameters:          {sum(p.numel() for p in winner_model.parameters()):,}")
    print()
    print("Test set, evaluated once on the chosen model:")
    print(f"  On-target Spearman: {winner['test_ontarget_spearman']:.4f}")
    print(f"  On-target Pearson:  {winner['test_ontarget_pearson']:.4f}")
    print(f"  Off-target AUPRC:   {winner['test_offtarget_auprc']:.4f}")
    print()
    print(f"Log saved to:  {LOG_PATH}")
    print(f"Plot saved to: {FIG_PATH}")


if __name__ == "__main__":
    run_search()