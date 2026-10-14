import json
import random
import itertools
import numpy as np
import torch
import torch.nn as nn

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.models.train_multitask import get_offtarget_loaders
from src.eval.metrics import regression_metrics, classification_metrics


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 456
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
ALPHA = 0.7

USE_LOSS_NORMALIZATION = True

NAS_LOG_PATH = "experiments/logs/nas_multitask_search_log.json"


# ============================================================
# ARCHITECTURES
# ============================================================

# ------------------------------------------------------------
# 1. ORIGINAL CNN BASELINE
#
# Verified against the real search_space.py via random.seed(7) +
# sample_architecture(). Historical result under THIS run's settings
# may differ slightly from any earlier session — that's expected,
# since training is stochastic even with a fixed architecture.
# ------------------------------------------------------------

CNN_BASELINE = [
    {"block_type": "cnn_large", "channels": 32, "dropout": 0.2},
    {"block_type": "cnn_small", "channels": 16, "dropout": 0.2},
    {"block_type": "cnn_small", "channels": 32, "dropout": 0.2},
    {"block_type": "cnn_small", "channels": 64, "dropout": 0.0},
]


# ------------------------------------------------------------
# 2. STRONG BILSTM BASELINE (fixed, hand-designed — not a NAS output)
# ------------------------------------------------------------

BILSTM_BASELINE = [
    {"block_type": "cnn_large", "channels": 64, "dropout": 0.1},
    {"block_type": "bilstm", "channels": 64, "dropout": 0.1},
]


# ------------------------------------------------------------
# 3. NAS WINNER — loaded directly from the real search log,
# not hand-typed, so there's no risk of a transcription mismatch
# (the earlier hand-typed version had dropout=0.1 on the BiLSTM
# layer; the real logged winner has dropout=0.2).
# ------------------------------------------------------------

def load_nas_winner(log_path=NAS_LOG_PATH):
    with open(log_path) as f:
        log = json.load(f)
    best = max(log["history"], key=lambda h: h["fitness"])
    return best["architecture"], best["fitness"]


NAS_WINNER, NAS_WINNER_LOGGED_FITNESS = load_nas_winner()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# TRAINING
# ============================================================

def train_model(model):
    ot_train = get_dataloader("data/processed/ontarget_splits.npz", "train", BATCH_SIZE, shuffle=True)
    ot_val = get_dataloader("data/processed/ontarget_splits.npz", "val", BATCH_SIZE, shuffle=False)
    off_train, off_val, off_test, pos_weight = get_offtarget_loaders(BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    running_ot = 1.0
    running_off = 1.0
    momentum = 0.9

    for epoch in range(EPOCHS):
        model.train()
        total_ot = 0.0
        total_off = 0.0
        n_batches = 0

        off_cycle = itertools.cycle(off_train)

        for (x_seq, x_aux, y_ot), (x_pair, y_off) in zip(ot_train, off_cycle):
            optimizer.zero_grad()

            pred_ot = model.forward_ontarget(x_seq, x_aux)
            pred_off = model.forward_offtarget(x_pair)

            loss_ot = mse_loss(pred_ot, y_ot)
            loss_off = bce_loss(pred_off, y_off)

            total_ot += loss_ot.item()
            total_off += loss_off.item()
            n_batches += 1

            if USE_LOSS_NORMALIZATION:
                running_ot = momentum * running_ot + (1.0 - momentum) * loss_ot.item()
                running_off = momentum * running_off + (1.0 - momentum) * loss_off.item()
                norm_loss_ot = loss_ot / (running_ot + 1e-8)
                norm_loss_off = loss_off / (running_off + 1e-8)
                combined_loss = ALPHA * norm_loss_ot + (1.0 - ALPHA) * norm_loss_off
            else:
                combined_loss = ALPHA * loss_ot + (1.0 - ALPHA) * loss_off

            combined_loss.backward()
            optimizer.step()

        model.eval()
        ot_predictions, ot_targets = [], []
        off_predictions, off_targets = [], []

        with torch.no_grad():
            for x_seq, x_aux, y_ot in ot_val:
                pred = model.forward_ontarget(x_seq, x_aux)
                ot_predictions.append(pred.cpu())
                ot_targets.append(y_ot.cpu())

            for x_pair, y_off in off_val:
                pred = model.forward_offtarget(x_pair)
                probability = torch.sigmoid(pred)
                off_predictions.append(probability.cpu())
                off_targets.append(y_off.cpu())

        ot_predictions = torch.cat(ot_predictions).numpy().reshape(-1)
        ot_targets = torch.cat(ot_targets).numpy().reshape(-1)
        off_predictions = torch.cat(off_predictions).numpy().reshape(-1)
        off_targets = torch.cat(off_targets).numpy().reshape(-1)

        ot_metrics = regression_metrics(ot_targets, ot_predictions)
        off_metrics = classification_metrics(off_targets, off_predictions)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Train OT loss:  {total_ot / n_batches:.4f}")
        print(f"  Train OFF loss: {total_off / n_batches:.4f}")
        print(f"  Val Spearman:   {ot_metrics['spearman']:.4f}")
        print(f"  Val AUPRC:      {off_metrics['auprc']:.4f}")
        print()


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(model):
    ot_test = get_dataloader("data/processed/ontarget_splits.npz", "test", BATCH_SIZE, shuffle=False)
    _, _, off_test, _ = get_offtarget_loaders(BATCH_SIZE)

    model.eval()
    ot_predictions, ot_targets = [], []
    off_predictions, off_targets = [], []

    with torch.no_grad():
        for x_seq, x_aux, y_ot in ot_test:
            pred = model.forward_ontarget(x_seq, x_aux)
            ot_predictions.append(pred.cpu())
            ot_targets.append(y_ot.cpu())

        for x_pair, y_off in off_test:
            pred = model.forward_offtarget(x_pair)
            probability = torch.sigmoid(pred)
            off_predictions.append(probability.cpu())
            off_targets.append(y_off.cpu())

    ot_predictions = torch.cat(ot_predictions).numpy().reshape(-1)
    ot_targets = torch.cat(ot_targets).numpy().reshape(-1)
    off_predictions = torch.cat(off_predictions).numpy().reshape(-1)
    off_targets = torch.cat(off_targets).numpy().reshape(-1)

    ot_metrics = regression_metrics(ot_targets, ot_predictions)
    off_metrics = classification_metrics(off_targets, off_predictions)

    fitness = (max(ot_metrics["spearman"], 0.0) + off_metrics["auprc"]) / 2.0

    return ot_metrics, off_metrics, fitness


# ============================================================
# RUN ONE MODEL
# ============================================================

def run_experiment(name, architecture):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print()
    print("Architecture:")
    for i, layer in enumerate(architecture, start=1):
        print(f"  Layer {i}: {layer}")

    set_seed(SEED)
    model = MultiTaskNASModel(architecture)
    parameter_count = sum(p.numel() for p in model.parameters())

    print()
    print(f"Number of parameters: {parameter_count:,}")
    print()

    train_model(model)
    ot_metrics, off_metrics, fitness = evaluate_test(model)

    print()
    print("FINAL TEST")
    print(f"  On-target Spearman: {ot_metrics['spearman']:.4f}")
    print(f"  On-target Pearson:  {ot_metrics['pearson']:.4f}")
    print(f"  Off-target AUPRC:    {off_metrics['auprc']:.4f}")
    print(f"  NAS fitness:        {fitness:.4f}")

    return {
        "name": name,
        "parameters": parameter_count,
        "ot_spearman": ot_metrics["spearman"],
        "ot_pearson": ot_metrics["pearson"],
        "off_auprc": off_metrics["auprc"],
        "fitness": fitness,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("FINAL CONTROLLED BASELINE COMPARISON")
    print("=" * 70)
    print()
    print(f"Seed:                 {SEED}")
    print(f"Epochs:               {EPOCHS}")
    print(f"Batch size:           {BATCH_SIZE}")
    print(f"Learning rate:        {LR}")
    print(f"Alpha:                {ALPHA}")
    print(f"Loss normalization:   {USE_LOSS_NORMALIZATION}")
    print()
    print(f"NAS winner loaded from: {NAS_LOG_PATH}")
    print(f"  Architecture: {[l['block_type'] for l in NAS_WINNER]}")
    print(f"  Logged search fitness: {NAS_WINNER_LOGGED_FITNESS:.4f}")

    results = []

    results.append(run_experiment("1. ORIGINAL CNN BASELINE", CNN_BASELINE))
    results.append(run_experiment("2. CNN + BILSTM BASELINE", BILSTM_BASELINE))
    results.append(run_experiment("3. NAS WINNER (from log)", NAS_WINNER))

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Model':35s} {'Params':>10s} {'OT':>10s} {'OFF':>10s} {'Fitness':>10s}")
    print("-" * 80)
    for result in results:
        print(f"{result['name']:35s} {result['parameters']:10,d} "
              f"{result['ot_spearman']:10.4f} {result['off_auprc']:10.4f} {result['fitness']:10.4f}")
    print()