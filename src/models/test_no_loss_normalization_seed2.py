import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics
from src.nas.search_space import sample_architecture


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.7
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# This is the ONLY thing we are changing from the previous run.
RANDOM_SEED = 123


# ============================================================
# OFF-TARGET DATA LOADER
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
            / max((data["y_train"] == 1).sum(), 1)
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
# MULTITASK TRAINING
# ============================================================

def train_multitask_no_normalization(
    model,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    alpha=0.7,
):
    # --------------------------------------------------------
    # ON-TARGET DATA
    # --------------------------------------------------------

    ot_train = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "train",
        batch_size,
        shuffle=True,
    )

    ot_val = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "val",
        batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # OFF-TARGET DATA
    # --------------------------------------------------------

    (
        off_train,
        off_val,
        off_test,
        pos_weight,
    ) = get_offtarget_loaders(batch_size)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # --------------------------------------------------------
    # LOSS FUNCTIONS
    # --------------------------------------------------------

    mse_loss = nn.MSELoss()

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_ot_loss = 0.0
        total_off_loss = 0.0
        total_combined_loss = 0.0
        n_steps = 0

        off_cycle = itertools.cycle(
            off_train
        )

        for (
            (x_seq, x_aux, y_ot),
            (x_pair, y_off),
        ) in zip(
            ot_train,
            off_cycle,
        ):

            optimizer.zero_grad()

            # On-target prediction
            pred_ot = model.forward_ontarget(
                x_seq,
                x_aux,
            )

            # Off-target prediction
            pred_off = model.forward_offtarget(
                x_pair
            )

            # Individual losses
            loss_ot = mse_loss(
                pred_ot,
                y_ot,
            )

            loss_off = bce_loss(
                pred_off,
                y_off,
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # NO dynamic loss normalization.
            #
            # Fixed multitask objective:
            #
            # 0.7 * OT + 0.3 * OFF
            # ------------------------------------------------

            loss = (
                alpha * loss_ot
                + (1.0 - alpha) * loss_off
            )

            loss.backward()

            optimizer.step()

            total_ot_loss += loss_ot.item()
            total_off_loss += loss_off.item()
            total_combined_loss += loss.item()

            n_steps += 1

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        ot_preds = []
        ot_trues = []

        off_preds = []
        off_trues = []

        with torch.no_grad():

            # On-target validation
            for x_seq, x_aux, y in ot_val:

                predictions = model.forward_ontarget(
                    x_seq,
                    x_aux,
                )

                ot_preds.extend(
                    predictions.numpy()
                )

                ot_trues.extend(
                    y.numpy()
                )

            # Off-target validation
            for x_pair, y in off_val:

                predictions = torch.sigmoid(
                    model.forward_offtarget(
                        x_pair
                    )
                )

                off_preds.extend(
                    predictions.numpy()
                )

                off_trues.extend(
                    y.numpy()
                )

        ot_metrics = regression_metrics(
            ot_trues,
            ot_preds,
        )

        off_metrics = classification_metrics(
            off_trues,
            off_preds,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"OT loss: "
            f"{total_ot_loss / n_steps:.4f}, "
            f"OFF loss: "
            f"{total_off_loss / n_steps:.4f}, "
            f"Combined loss: "
            f"{total_combined_loss / n_steps:.4f} | "
            f"Val Spearman: "
            f"{ot_metrics['spearman']:.4f}, "
            f"Val AUPRC: "
            f"{off_metrics['auprc']:.4f}"
        )

    # ========================================================
    # FINAL TEST
    # ========================================================

    model.eval()

    ot_test_loader = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "test",
        batch_size,
        shuffle=False,
    )

    ot_preds = []
    ot_trues = []

    off_preds = []
    off_trues = []

    with torch.no_grad():

        # On-target test
        for x_seq, x_aux, y in ot_test_loader:

            predictions = model.forward_ontarget(
                x_seq,
                x_aux,
            )

            ot_preds.extend(
                predictions.numpy()
            )

            ot_trues.extend(
                y.numpy()
            )

        # Off-target test
        for x_pair, y in off_test:

            predictions = torch.sigmoid(
                model.forward_offtarget(
                    x_pair
                )
            )

            off_preds.extend(
                predictions.numpy()
            )

            off_trues.extend(
                y.numpy()
            )

    ot_test_metrics = regression_metrics(
        ot_trues,
        ot_preds,
    )

    off_test_metrics = classification_metrics(
        off_trues,
        off_preds,
    )

    print()
    print("=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(
        f"On-target Spearman: "
        f"{ot_test_metrics['spearman']:.4f}"
    )

    print(
        f"On-target Pearson:  "
        f"{ot_test_metrics['pearson']:.4f}"
    )

    print(
        f"Off-target AUPRC:   "
        f"{off_test_metrics['auprc']:.4f}"
    )

    print("=" * 65)

    return (
        ot_test_metrics,
        off_test_metrics,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Seed
    #
    # IMPORTANT:
    # We deliberately use a different seed from the previous
    # experiment.
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    arch = sample_architecture()

    print(
        "Random seed:",
        RANDOM_SEED,
    )

    print(
        "Sampled architecture:",
        [
            layer["block_type"]
            for layer in arch
        ],
    )

    # --------------------------------------------------------
    # PyTorch seed
    # --------------------------------------------------------

    torch.manual_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MultiTaskNASModel(
        arch
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    print()
    print(
        "Testing multitask training WITHOUT "
        "dynamic loss normalization"
    )

    print(
        f"Alpha = {ALPHA:.2f}"
    )

    print(
        f"Epochs = {EPOCHS}"
    )

    print(
        f"Batch size = {BATCH_SIZE}"
    )

    print(
        f"Learning rate = {LEARNING_RATE}"
    )

    print()

    train_multitask_no_normalization(
        model=model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        alpha=ALPHA,
    )
