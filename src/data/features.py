# src/data/features.py
import RNA  # ViennaRNA Python binding


def gc_content(seq: str) -> float:
    """Fraction of G/C bases in the sequence."""
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def melting_temperature(seq: str) -> float:
    """Simple Wallace rule approximation."""
    seq = seq.upper()
    at = seq.count("A") + seq.count("T")
    gc = seq.count("G") + seq.count("C")
    return 2 * at + 4 * gc


def mfe_secondary_structure(seq: str) -> float:
    """Minimum free energy via ViennaRNA. Converts DNA (T) to RNA (U) first,
    since ViennaRNA expects RNA sequences."""
    rna_seq = seq.upper().replace("T", "U")
    structure, mfe = RNA.fold(rna_seq)
    return mfe


def compute_all_features(seq: str) -> dict:
    return {
        "gc_content": gc_content(seq),
        "melting_temp": melting_temperature(seq),
        "mfe": mfe_secondary_structure(seq),
    }


def build_feature_dataframe(df, seq_col="sequence"):
    """Compute GC/Tm/MFE for every sequence in the dataframe and attach as new columns."""
    import pandas as pd
    from tqdm import tqdm

    records = []
    for seq in tqdm(df[seq_col], desc="Computing features"):
        records.append(compute_all_features(seq))

    feat_df = pd.DataFrame(records)
    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return result


if __name__ == "__main__":
    from src.data.download import get_clean_ontarget_df

    df = get_clean_ontarget_df()

    print("Testing feature extraction on 10 sample sequences:\n")
    sample_seqs = df["sequence"].head(10).tolist()
    for seq in sample_seqs:
        feats = compute_all_features(seq)
        print(f"{seq}: GC={feats['gc_content']:.3f}, Tm={feats['melting_temp']}, MFE={feats['mfe']:.2f}")

    print("\n--- Now processing the full dataset (this will take a while) ---")
    full_df = build_feature_dataframe(df)

    print(full_df.head())
    full_df.to_csv("data/processed/ontarget_with_features.csv", index=False)
    print(f"\nSaved {len(full_df)} rows with features to data/processed/ontarget_with_features.csv")