# src/data/preprocess.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """23-mer -> (23, 4) one-hot matrix."""
    seq = seq.upper()
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in BASE_TO_IDX:
            encoded[i, BASE_TO_IDX[base]] = 1.0
        # unexpected characters (e.g. 'N') stay as an all-zero row
    return encoded


def encode_all_sequences(df, seq_col="sequence") -> np.ndarray:
    """Encode every sequence in the dataframe into a stacked (N, 23, 4) array."""
    return np.stack([one_hot_encode_sequence(s) for s in df[seq_col]])


def scale_auxiliary_features(df, feature_cols=("gc_content", "melting_temp", "mfe")):
    """Standardize the biological features (mean 0, std 1)."""
    feature_cols = list(feature_cols)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in feature_cols])
    return scaled_df, scaler


def split_dataset(X_seq, X_aux, y, train_frac=0.765, val_frac=0.085, test_frac=0.15, seed=42):
    """Split sequence arrays, auxiliary features, and labels together, consistently."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    # First split: train vs (val + test)
    X_seq_train, X_seq_temp, X_aux_train, X_aux_temp, y_train, y_temp = train_test_split(
        X_seq, X_aux, y, test_size=(val_frac + test_frac), random_state=seed
    )

    # Second split: val vs test, out of the temp portion
    relative_val = val_frac / (val_frac + test_frac)
    X_seq_val, X_seq_test, X_aux_val, X_aux_test, y_val, y_test = train_test_split(
        X_seq_temp, X_aux_temp, y_temp, test_size=(1 - relative_val), random_state=seed
    )

    return {
        "X_seq_train": X_seq_train, "X_aux_train": X_aux_train, "y_train": y_train,
        "X_seq_val": X_seq_val, "X_aux_val": X_aux_val, "y_val": y_val,
        "X_seq_test": X_seq_test, "X_aux_test": X_aux_test, "y_test": y_test,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/processed/ontarget_with_features.csv")
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    print("\nEncoding all sequences...")
    X_seq = encode_all_sequences(df)
    print(f"Full encoded array shape: {X_seq.shape}")

    print("\nScaling auxiliary features...")
    scaled_df, scaler = scale_auxiliary_features(df)
    X_aux = scaled_df.values
    y = df["efficiency"].values

    print("\nSplitting into train/val/test...")
    splits = split_dataset(X_seq, X_aux, y)
    for key, arr in splits.items():
        print(f"{key}: shape {arr.shape}")

    np.savez("data/processed/ontarget_splits.npz", **splits)
    print("\nSaved splits to data/processed/ontarget_splits.npz")