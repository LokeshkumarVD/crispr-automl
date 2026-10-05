# src/data/split_offtarget.py
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42


def split_and_save(npz_in, npz_out, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """Split off-target data into train/val/test, stratified to preserve the
    positive/negative ratio in every split (critical given the severe imbalance)."""
    data = np.load(npz_in)
    X, y = data["X"], data["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_frac + test_frac), random_state=SEED, stratify=y
    )
    relative_val = val_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=SEED, stratify=y_temp
    )

    np.savez(
        npz_out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    print(f"{npz_out}:")
    print(f"  train: {len(y_train)} ({(y_train==1).sum()} positive)")
    print(f"  val:   {len(y_val)} ({(y_val==1).sum()} positive)")
    print(f"  test:  {len(y_test)} ({(y_test==1).sum()} positive)")


if __name__ == "__main__":
    split_and_save(
        "data/processed/offtarget_crispr.npz",
        "data/processed/offtarget_crispr_splits.npz",
    )