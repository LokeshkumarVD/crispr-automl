# src/models/test_nas_winner_seed123.py

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics


# ============================================================
# CONTROLLED SETTINGS
# ============================================================

SEED = 123

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ALPHA = 0.7


# ============================================================
# NAS-DISCOVERED ARCHITECTURE
# ============================================================

ARCHITECTURE = [
    {
        "block_type": "bilstm",
        "channels": 64,
        "dropout": 0.1,
    },
    {
        "block_type": "attention",
        "channels": 16,
        "dropout": 0.1,
        "num_heads": 8,
    },
]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# OFF-TARGET DATA
# ============================================================

def get_offtarget_loaders(batch_size=32):

    data = np.load(
        "data/processed/offtarget_crispr_splits.npz"
    )

    def make(split, shuffle):

        ds = TensorDataset(
            torch.tensor(
                data[f"X_{split}"],
                dtype=torch.float32,
            ),
            torch.tensor(
                data[f"y_{split}"],
                dtype=torch.float32,
            ),
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    pos_weight = torch.tensor(
        [
            (data["y_train"] == 0).sum()
            / max(
                (data["y_train"] == 1).sum(),
                1,
            )
        ],
        dtype=torch.float32,
    )

    return (
        make("train", True),
        make("val", False),
        make("test", False),
        pos_weight,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    ot_loader,
    off_loader,
):

    model.eval()

    # --------------------------------------------------------
    # On-target
    # --------------------------------------------------------

    ot_true = []
    ot_pred = []

    for x_seq, x_aux, y in ot_loader:

        pred = model.forward_ontarget(
            x_seq,
            x_aux,
        )

        ot_true.extend(
            y.cpu().numpy()
        )

        ot_pred.extend(
            pred.cpu().numpy()
        )

    ot_true = np.asarray(
        ot_true
    ).reshape(-1)

    ot_pred = np.asarray(
        ot_pred
    ).reshape(-1)

    ot_metrics = regression_metrics(
        ot_true,
        ot_pred,
    )

    # --------------------------------------------------------
    # Off-target
    # --------------------------------------------------------

    off_true = []
    off_prob = []

    for x_pair, y in off_loader:

        logits = model.forward_offtarget(
            x_pair
        )

        prob = torch.sigmoid(
            logits
        )

        off_true.extend(
            y.cpu().numpy()
        )

        off_prob.extend(
            prob.cpu().numpy()
        )

    off_true = np.asarray(
        off_true
    ).reshape(-1)

    off_prob = np.asarray(
        off_prob
    ).reshape(-1)

    off_metrics = classification_metrics(
        off_true,
        off_prob,
    )

    return (
        ot_metrics,
        off_metrics,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model():

    set_seed(SEED)

    print("====================================================")
    print("NAS WINNER VALIDATION")
    print("====================================================")
    print(f"Seed: {SEED}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Alpha: {ALPHA}")
    print()
    print("Architecture:")
    print("  Layer 1: bilstm, channels=64, dropout=0.1")
    print(
        "  Layer 2: attention, channels=16, "
        "dropout=0.1, heads=8"
    )
    print()
    print("Dynamic loss normalization: ON")
    print("====================================================")
    print()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    ot_train = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "train",
        BATCH_SIZE,
        shuffle=True,
    )

    ot_val = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "val",
        BATCH_SIZE,
        shuffle=False,
    )

    ot_test = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "test",
        BATCH_SIZE,
        shuffle=False,
    )

    (
        off_train,
        off_val,
        off_test,
        pos_weight,
    ) = get_offtarget_loaders(
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiTaskNASModel(
        ARCHITECTURE
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Number of parameters: {n_params:,}"
    )
    print()

    # --------------------------------------------------------
    # Optimizer and losses
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    mse_loss = nn.MSELoss()

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Dynamic normalization
    # --------------------------------------------------------

    running_ot = 1.0
    running_off = 1.0

    momentum = 0.9

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    import itertools

    for epoch in range(EPOCHS):

        model.train()

        epoch_ot_loss = 0.0
        epoch_off_loss = 0.0
        n_batches = 0

        off_cycle = itertools.cycle(
            off_train
        )

        for (
            x_seq,
            x_aux,
            y_ot,
        ), (
            x_pair,
            y_off,
        ) in zip(
            ot_train,
            off_cycle,
        ):

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward passes
            # ------------------------------------------------

            pred_ot = model.forward_ontarget(
                x_seq,
                x_aux,
            )

            pred_off = model.forward_offtarget(
                x_pair
            )

            # ------------------------------------------------
            # Individual losses
            # ------------------------------------------------

            loss_ot = mse_loss(
                pred_ot,
                y_ot,
            )

            loss_off = bce_loss(
                pred_off,
                y_off,
            )

            # ------------------------------------------------
            # Update running loss estimates
            # ------------------------------------------------

            running_ot = (
                momentum * running_ot
                + (1 - momentum)
                * loss_ot.item()
            )

            running_off = (
                momentum * running_off
                + (1 - momentum)
                * loss_off.item()
            )

            # ------------------------------------------------
            # Normalize losses
            # ------------------------------------------------

            norm_loss_ot = (
                loss_ot
                / (running_ot + 1e-8)
            )

            norm_loss_off = (
                loss_off
                / (running_off + 1e-8)
            )

            # ------------------------------------------------
            # Final multitask loss
            # ------------------------------------------------

            loss = (
                ALPHA * norm_loss_ot
                + (1 - ALPHA)
                * norm_loss_off
            )

            loss.backward()

            optimizer.step()

            epoch_ot_loss += loss_ot.item()
            epoch_off_loss += loss_off.item()

            n_batches += 1

        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        avg_ot_loss = (
            epoch_ot_loss / n_batches
        )

        avg_off_loss = (
            epoch_off_loss / n_batches
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_ot_metrics, val_off_metrics = evaluate(
            model,
            ot_val,
            off_val,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"  Train OT loss:  "
            f"{avg_ot_loss:.4f}"
        )

        print(
            f"  Train OFF loss: "
            f"{avg_off_loss:.4f}"
        )

        print(
            f"  Running OT:     "
            f"{running_ot:.6f}"
        )

        print(
            f"  Running OFF:    "
            f"{running_off:.6f}"
        )

        print(
            f"  Val Spearman:   "
            f"{val_ot_metrics['spearman']:.4f}"
        )

        print(
            f"  Val AUPRC:      "
            f"{val_off_metrics['auprc']:.4f}"
        )

        print()

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_ot_metrics, test_off_metrics = evaluate(
        model,
        ot_test,
        off_test,
    )

    print("====================================================")
    print("FINAL TEST")
    print("====================================================")

    print(
        f"On-target Spearman: "
        f"{test_ot_metrics['spearman']:.4f}"
    )

    print(
        f"On-target Pearson:  "
        f"{test_ot_metrics['pearson']:.4f}"
    )

    print(
        f"Off-target AUPRC:   "
        f"{test_off_metrics['auprc']:.4f}"
    )

    fitness = (
        max(
            test_ot_metrics["spearman"],
            0.0,
        )
        + test_off_metrics["auprc"]
    ) / 2

    print(
        f"NAS fitness:        "
        f"{fitness:.4f}"
    )

    print("====================================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()