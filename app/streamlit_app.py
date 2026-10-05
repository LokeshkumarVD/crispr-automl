# app/streamlit_app.py
import sys
from pathlib import Path

# Make sure src/ is importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.data.features import compute_all_features
from src.data.preprocess import one_hot_encode_sequence
from src.models.nas_model import NASModel, AttentionLayer

CKPT_PATH = "experiments/checkpoints/nas_best.pt"
FEATURES_CSV = "data/processed/ontarget_with_features.csv"


@st.cache_resource
def load_model_and_scaler():
    """Load the saved NAS model, and refit the scaler on the same training
    data/columns used originally (the scaler object itself wasn't saved in Phase 1)."""
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    model = NASModel(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    df = pd.read_csv(FEATURES_CSV)
    scaler = StandardScaler()
    scaler.fit(df[["gc_content", "melting_temp", "mfe"]])

    return model, scaler, checkpoint["architecture"]


def validate_sequence(seq: str) -> str | None:
    """Return an error message if invalid, else None."""
    seq = seq.upper().strip()
    if len(seq) != 23:
        return f"Sequence must be exactly 23 characters (20nt + NGG PAM). Got {len(seq)}."
    if not set(seq).issubset({"A", "C", "G", "T"}):
        return "Sequence must contain only A, C, G, T."
    if seq[-2:] != "GG":
        return "Warning: the last 2 characters should be 'GG' for a valid NGG PAM (prediction will still run)."
    return None


def predict(seq: str, model, scaler):
    seq = seq.upper().strip()
    feats = compute_all_features(seq)
    raw_aux = np.array([[feats["gc_content"], feats["melting_temp"], feats["mfe"]]])
    scaled_aux = scaler.transform(raw_aux)[0]

    x_seq = torch.tensor(one_hot_encode_sequence(seq), dtype=torch.float32).unsqueeze(0)
    x_aux = torch.tensor(scaled_aux, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x_seq, x_aux).item()
    pred = max(0.0, min(1.0, pred))  # clip for display, since the model can output slightly outside [0,1]

    return pred, feats


def get_attention_weights(model):
    """Pull attention weights from any AttentionLayer in the model, if present."""
    for layer in model.backbone:
        if isinstance(layer, AttentionLayer) and layer.last_attn is not None:
            return layer.last_attn[0].numpy()  # (seq_len, seq_len), batch size 1
    return None


st.set_page_config(page_title="CRISPR sgRNA Efficiency Predictor", layout="centered")
st.title("CRISPR sgRNA On-Target Efficiency Predictor")
st.caption("NAS-discovered architecture — trained on DeepHF on-target efficiency data")

try:
    model, scaler, architecture = load_model_and_scaler()
except FileNotFoundError:
    st.error(
        f"Could not find `{CKPT_PATH}`. Run `python -m src.nas.final_train` first "
        "to train and save the model."
    )
    st.stop()

with st.expander("Model architecture used"):
    st.write(architecture)

seq_input = st.text_input(
    "Enter a 23-nucleotide sgRNA sequence (20nt protospacer + NGG PAM)",
    value="AAAAAAAAACTCCAAAACCCTGG",
    max_chars=23,
)

if st.button("Predict efficiency", type="primary"):
    error = validate_sequence(seq_input)
    if error and "Warning" not in error:
        st.error(error)
    else:
        if error:
            st.warning(error)

        pred, feats = predict(seq_input, model, scaler)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted efficiency", f"{pred:.3f}")
            st.progress(pred)
        with col2:
            st.write("**Biological features**")
            st.write(f"GC content: {feats['gc_content']:.3f}")
            st.write(f"Melting temp: {feats['melting_temp']}")
            st.write(f"MFE: {feats['mfe']:.2f}")

        attn = get_attention_weights(model)
        if attn is not None:
            st.write("**Attention pattern** (which positions the model focused on)")
            per_position = attn.mean(axis=0)  # average attention received by each position
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.bar(range(len(per_position)), per_position)
            ax.set_xlabel("Sequence position (0-22)")
            ax.set_ylabel("Avg. attention received")
            ax.set_title("Which sequence positions the model attended to most")
            st.pyplot(fig)
        else:
            st.caption("This architecture has no attention layer, so no attention map is shown.")

st.divider()
st.caption(
    "Research prototype — architecture selected via a small-scale evolutionary NAS search, "
    "trained on the DeepHF on-target efficiency dataset. Not for clinical or production use."
)