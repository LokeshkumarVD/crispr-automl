# src/data/data_report.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_ontarget_stats():
    """Sequence length + efficiency distribution + feature ranges, for on-target data."""
    df = pd.read_csv("data/processed/ontarget_with_features.csv")

    # Sequence length check (should be a single spike at 23 if data is clean)
    seq_lengths = df["sequence"].str.len()
    plt.figure()
    plt.hist(seq_lengths, bins=10)
    plt.title("On-target: sequence length distribution")
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.savefig(FIGURES_DIR / "ontarget_seq_length_dist.png")
    plt.close()

    # Efficiency score distribution
    plt.figure()
    plt.hist(df["efficiency"], bins=30)
    plt.title("On-target: efficiency score distribution")
    plt.xlabel("Efficiency")
    plt.ylabel("Count")
    plt.savefig(FIGURES_DIR / "ontarget_efficiency_dist.png")
    plt.close()

    # Auxiliary feature ranges
    plt.figure()
    plt.boxplot(
        [df["gc_content"], df["melting_temp"], df["mfe"]],
        tick_labels=["GC content", "Melting temp", "MFE"],
    )
    plt.title("On-target: auxiliary feature ranges")
    plt.savefig(FIGURES_DIR / "ontarget_feature_ranges.png")
    plt.close()

    print(f"On-target: {len(df)} samples, sequence lengths: {seq_lengths.unique()}")


def plot_offtarget_class_balance():
    """Class balance bar chart for both off-target datasets."""
    crispr_data = np.load("data/processed/offtarget_crispr.npz")
    guideseq_data = np.load("data/processed/offtarget_guideseq.npz")

    for name, data in [("crispr", crispr_data), ("guideseq", guideseq_data)]:
        y = data["y"]
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()

        plt.figure()
        plt.bar(["Negative", "Positive"], [n_neg, n_pos])
        plt.title(f"Off-target class balance: {name}")
        plt.ylabel("Count")
        plt.savefig(FIGURES_DIR / f"offtarget_{name}_class_balance.png")
        plt.close()

        print(f"{name}: {n_neg} negative, {n_pos} positive ({100 * n_pos / (n_pos + n_neg):.2f}% positive)")


if __name__ == "__main__":
    print("Generating on-target statistics...")
    plot_ontarget_stats()

    print("\nGenerating off-target class balance plots...")
    plot_offtarget_class_balance()

    print(f"\nAll figures saved to {FIGURES_DIR}/")