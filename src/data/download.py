# src/data/download.py
import pickle
import pandas as pd
from pathlib import Path
import numpy as np

EXTERNAL = Path("data/external_repos")
OFFTARGET_PKL = Path("data/raw/offtarget/encoded_data_charlier_et_al.pkl")


# ============================================================
# OFF-TARGET DATA (Charlier et al. 2021 encoding, dagrate repo)
# ============================================================

def load_offtarget_data():
    """
    Load the Charlier et al. 2021 encoded off-target data.
    This is a DICTIONARY of multiple sklearn Bunch objects, one per dataset.
    """
    with open(OFFTARGET_PKL, "rb") as f:
        data = pickle.load(f)
    print("Available off-target datasets:")
    for key in data.keys():
        print(f"  - {key}")
    return data


def get_offtarget_bunch(data_dict, dataset_key):
    """Pull out one dataset's X/y arrays. This Bunch uses sklearn's image-dataset
    convention: features are under 'images', not 'data'."""
    bunch = data_dict[dataset_key]
    X = bunch.images
    y = bunch.target
    print(f"'{dataset_key}': X shape {X.shape}, y shape {y.shape}")
    print(f"  target_names: {list(bunch.target_names)}")
    return X, y


# ============================================================
# ON-TARGET DATA (DeepHF efficiency data, via CRISPRpredSEQ repo)
# ============================================================

def load_ontarget_xlsx():
    """Load DeepHF's on-target efficiency data (continuous scores, 3 Cas9 variants)."""
    path = EXTERNAL / "CRISPRpredSEQ" / "deephf_data.xlsx"
    df = pd.read_excel(path, sheet_name="Sheet1", header=1)
    print(f"Loaded on-target XLSX: {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


def get_clean_ontarget_df():
    """Load and clean the on-target dataset: drop missing labels, keep only what's needed."""
    df = load_ontarget_xlsx()
    original_len = df.shape[0]
    df = df.dropna(subset=["Wt_Efficiency"]).reset_index(drop=True)
    df = df[["sgRNA", "Wt_Efficiency", "eSpCas 9_Efficiency", "SpCas9-HF1_Efficiency"]].copy()
    df = df.rename(columns={"sgRNA": "sequence", "Wt_Efficiency": "efficiency"})
    print(f"Cleaned dataset: {df.shape[0]} rows (dropped {original_len - df.shape[0]} with missing efficiency)")
    return df

def save_offtarget_arrays():
    """Save both off-target datasets to disk as .npz, so they don't need reloading
    from the pickle every time."""
    data_dict = load_offtarget_data()

    X_crispr, y_crispr = get_offtarget_bunch(data_dict, "encodedDataCrispr")
    np.savez("data/processed/offtarget_crispr.npz", X=X_crispr, y=y_crispr)

    X_guideseq, y_guideseq = get_offtarget_bunch(data_dict, "encodedDataGuideSeq")
    np.savez("data/processed/offtarget_guideseq.npz", X=X_guideseq, y=y_guideseq)

    print("Saved both off-target datasets to data/processed/")
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ON-TARGET DATA")
    print("=" * 60)

    ot_df = load_ontarget_xlsx()
    print("\nFirst 10 rows (key columns):")
    print(ot_df[["sgRNA", "PAM", "21mer", "Wt_Efficiency"]].head(10))
    print("\nSequence length checks:")
    print("sgRNA lengths:", ot_df["sgRNA"].str.len().unique())
    print("21mer lengths:", ot_df["21mer"].str.len().unique())
    print("PAM lengths:", ot_df["PAM"].str.len().unique())
    print("\nWt_Efficiency stats:")
    print(ot_df["Wt_Efficiency"].describe())
    print("\nMissing values in key columns:")
    print(ot_df[["sgRNA", "PAM", "21mer", "Wt_Efficiency"]].isna().sum())

    print("\n--- Now testing the cleaned version ---")
    clean_df = get_clean_ontarget_df()
    print(clean_df.head())
    print("Sequence length check on cleaned data:", clean_df["sequence"].str.len().unique())

    print("\n" + "=" * 60)
    print("OFF-TARGET DATA")
    print("=" * 60)

    offtarget_dict = load_offtarget_data()

    print("\n--- Inspecting both off-target datasets ---")
    X_crispr, y_crispr = get_offtarget_bunch(offtarget_dict, "encodedDataCrispr")
    print("Class balance (encodedDataCrispr):")
    print(pd.Series(y_crispr).value_counts())

    X_guideseq, y_guideseq = get_offtarget_bunch(offtarget_dict, "encodedDataGuideSeq")
    print("\nClass balance (encodedDataGuideSeq):")
    print(pd.Series(y_guideseq).value_counts())