import random
import numpy as np
import torch
import torch.nn as nn
import itertools

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# EXPERIMENT SETTINGS
# =========================================================

RANDOM_SEED = 456

ALPHA = 0.70
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# =========================================================
# EXACT ARCHITECTURE SAMPLED WITH SEED 123
# =========================================================

ARCHITECTURE = [
    {
        "block_type": "bilstm",
        "channels": 16,
        "dropout": 0.1,
    },
    {
        "block_type": "bilstm",
        "channels": 16,
        "dropout": 0.0,
    },
]


# =========================================================
# OFF-TARGET DATA LOADER
# Same implementation as train_multitask.py
# =========================================================

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


# =========================================================
# TRAINING
# =========================================================

def train_multitask(
    model,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    alpha=0.70,
):

    # -----------------------------------------------------
    # ON-TARGET LOADERS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # OFF-TARGET LOADERS
    # -----------------------------------------------------

    off_train, off_val, off_test, pos_weight = (
        get_offtarget_loaders(batch_size)
    )

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # -----------------------------------------------------
    # LOSSES
    # -----------------------------------------------------

    mse_loss = nn.MSELoss()

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # -----------------------------------------------------
    # DYNAMIC LOSS NORMALIZATION
    # EXACTLY AS ORIGINAL
    # -----------------------------------------------------

    running_ot = 1.0
    running_off = 1.0
    momentum = 0.9

    # =====================================================
    # TRAINING
    # =====================================================

    for epoch in range(epochs):

        model.train()

        total_ot_loss = 0.0
        total_off_loss = 0.0
        n_steps = 0

        off_cycle = itertools.cycle(off_train)

        for (x_seq, x_aux, y_ot), (x_pair, y_off) in zip(
            ot_train,
            off_cycle,
        ):

            # -------------------------------------------------
            # FORWARD
            # -------------------------------------------------

            optimizer.zero_grad()

            pred_ot = model.forward_ontarget(
                x_seq,
                x_aux,
            )

            pred_off = model.forward_offtarget(
                x_pair,
            )

            # -------------------------------------------------
            # RAW LOSSES
            # -------------------------------------------------

            loss_ot = mse_loss(
                pred_ot,
                y_ot,
            )

            loss_off = bce_loss(
                pred_off,
                y_off,
            )

            # -------------------------------------------------
            # UPDATE RUNNING LOSS MAGNITUDES
            # -------------------------------------------------

            running_ot = (
                momentum * running_ot
                + (1 - momentum) * loss_ot.item()
            )

            running_off = (
                momentum * running_off
                + (1 - momentum) * loss_off.item()
            )

            # -------------------------------------------------
            # NORMALIZE
            # -------------------------------------------------

            norm_loss_ot = loss_ot / (
                running_ot + 1e-8
            )

            norm_loss_off = loss_off / (
                running_off + 1e-8
            )

            # -------------------------------------------------
            # MULTITASK LOSS
            # -------------------------------------------------

            loss = (
                alpha * norm_loss_ot
                + (1 - alpha) * norm_loss_off
            )

            # -------------------------------------------------
            # BACKPROP
            # -------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # -------------------------------------------------
            # REPORTING
            # -------------------------------------------------

            total_ot_loss += loss_ot.item()
            total_off_loss += loss_off.item()

            n_steps += 1

        # =====================================================
        # TRAINING SUMMARY
        # =====================================================

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"  Train OT loss: "
            f"{total_ot_loss / n_steps:.4f}"
        )

        print(
            f"  Train OFF loss: "
            f"{total_off_loss / n_steps:.4f}"
        )

        # =====================================================
        # VALIDATION
        # =====================================================

        model.eval()

        ot_preds = []
        ot_trues = []

        off_preds = []
        off_trues = []

        with torch.no_grad():

            # -----------------------------
            # ON-TARGET
            # -----------------------------

            for x_seq, x_aux, y in ot_val:

                ot_preds.extend(
                    model.forward_ontarget(
                        x_seq,
                        x_aux,
                    ).numpy()
                )

                ot_trues.extend(
                    y.numpy()
                )

            # -----------------------------
            # OFF-TARGET
            # -----------------------------

            for x_pair, y in off_val:

                off_preds.extend(
                    torch.sigmoid(
                        model.forward_offtarget(
                            x_pair
                        )
                    ).numpy()
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
            f"  Val Spearman: "
            f"{ot_metrics['spearman']:.4f}"
        )

        print(
            f"  Val AUPRC: "
            f"{off_metrics['auprc']:.4f}"
        )

    # =====================================================
    # FINAL TEST
    # =====================================================

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

        # -----------------------------
        # ON-TARGET TEST
        # -----------------------------

        for x_seq, x_aux, y in ot_test_loader:

            ot_preds.extend(
                model.forward_ontarget(
                    x_seq,
                    x_aux,
                ).numpy()
            )

            ot_trues.extend(
                y.numpy()
            )

        # -----------------------------
        # OFF-TARGET TEST
        # -----------------------------

        for x_pair, y in off_test:

            off_preds.extend(
                torch.sigmoid(
                    model.forward_offtarget(
                        x_pair
                    )
                ).numpy()
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

    print("\nFINAL TEST RESULTS")

    print(
        f"On-target Spearman: "
        f"{ot_test_metrics['spearman']:.4f}"
    )

    print(
        f"On-target Pearson: "
        f"{ot_test_metrics['pearson']:.4f}"
    )

    print(
        f"Off-target AUPRC: "
        f"{off_test_metrics['auprc']:.4f}"
    )

    return (
        ot_test_metrics,
        off_test_metrics,
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        "Fixed architecture: "
        "['bilstm', 'bilstm']"
    )

    print(
        "Layer 1: bilstm, channels=16, dropout=0.1"
    )

    print(
        "Layer 2: bilstm, channels=16, dropout=0.0"
    )

    print(
        "\nTesting exact seed-123 architecture "
        "with seed 456"
    )

    print(
        "WITH dynamic loss normalization"
    )

    print(
        f"Alpha = {ALPHA}"
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

    model = MultiTaskNASModel(
        ARCHITECTURE
    )

    train_multitask(
        model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        alpha=ALPHA,
    )