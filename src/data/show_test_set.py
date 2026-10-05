# src/data/show_test_set.py
import numpy as np
import pandas as pd

IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


def decode_sequence(one_hot_seq: np.ndarray) -> str:
    """Reverse a (23, 4) one-hot array back into a DNA string."""
    indices = np.argmax(one_hot_seq, axis=1)
    return "".join(IDX_TO_BASE[i] for i in indices)


if __name__ == "__main__":
    data = np.load("data/processed/ontarget_splits.npz")
    X_seq_test = data["X_seq_test"]
    X_aux_test = data["X_aux_test"]
    y_test = data["y_test"]

    print(f"Test set size: {len(y_test)} samples\n")

    rows = []
    for i in range(len(y_test)):
        rows.append({
            "sequence": decode_sequence(X_seq_test[i]),
            "true_efficiency": round(float(y_test[i]), 4),
            "gc_scaled": round(float(X_aux_test[i, 0]), 3),
            "tm_scaled": round(float(X_aux_test[i, 1]), 3),
            "mfe_scaled": round(float(X_aux_test[i, 2]), 3),
        })

    test_df = pd.DataFrame(rows)
    print("First 10 rows of the held-out test set:\n")
    print(test_df.head(10).to_string(index=False))

    test_df.to_csv("data/processed/ontarget_test_set_readable.csv", index=False)
    print(f"\nFull readable test set saved to data/processed/ontarget_test_set_readable.csv")