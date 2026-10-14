import itertools
import random

import numpy as np
import torch
import torch.nn as nn

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics
from src.nas.search_space import sample_architecture


def get_offtarget_loaders(batch_size=32):
    data = np.load("data/processed/offtarget_crispr_splits.npz")

    def make(split, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(data[f"X_{split}"], dtype=torch.float32),
            torch.tensor(data[f"y_{split}"], dtype=torch.float32),
        )

        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    pos_weight = torch.tensor(
        [(data["y_train"] == 0).sum() /
         max((data["y_train"] == 1).sum(), 1)],
        dtype=torch.float32,
    )

    return (
        make("train", True),
        make("val", False),
        make("test", False),
        pos_weight,
    )


def train_multitask(
    model,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    alpha=0.7,
):
    """
    Multitask training with a specified alpha.

    alpha:
        Weight applied to on-target loss.

    1-alpha:
        Weight applied to off-target loss.

    Everything else is kept identical to the original
    multitask training setup.
    """

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
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

    off_train, off_val, off_test, pos_weight = (
        get_offtarget_loaders(batch_size)
    )

    # ---------------------------------------------------------
    # Optimizer and losses
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    mse_loss = nn.MSELoss()

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # Keep the same running-loss normalization used previously.
    running_ot = 1.0
    running_off = 1.0
    momentum = 0.9

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
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

            optimizer.zero_grad()

            # Forward passes
            pred_ot = model.forward_ontarget(
                x_seq,
                x_aux,
            )

            pred_off = model.forward_offtarget(
                x_pair,
            )

            # Raw task losses
            loss_ot = mse_loss(
                pred_ot,
                y_ot,
            )

            loss_off = bce_loss(
                pred_off,
                y_off,
            )

            # -------------------------------------------------
            # Running loss normalization
            # -------------------------------------------------
            running_ot = (
                momentum * running_ot
                + (1 - momentum) * loss_ot.item()
            )

            running_off = (
                momentum * running_off
                + (1 - momentum) * loss_off.item()
            )

            norm_loss_ot = (
                loss_ot / (running_ot + 1e-8)
            )

            norm_loss_off = (
                loss_off / (running_off + 1e-8)
            )

            # -------------------------------------------------
            # Weighted multitask loss
            # -------------------------------------------------
            loss = (
                alpha * norm_loss_ot
                + (1 - alpha) * norm_loss_off
            )

            loss.backward()

            optimizer.step()

            total_ot_loss += loss_ot.item()
            total_off_loss += loss_off.item()
            n_steps += 1

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        model.eval()

        ot_preds = []
        ot_trues = []

        off_preds = []
        off_trues = []

        with torch.no_grad():

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

            for x_pair, y in off_val:

                predictions = torch.sigmoid(
                    model.forward_offtarget(x_pair)
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
            f"OT loss: {total_ot_loss / n_steps:.4f}, "
            f"OFF loss: {total_off_loss / n_steps:.4f} | "
            f"Val Spearman: {ot_metrics['spearman']:.4f}, "
            f"Val AUPRC: {off_metrics['auprc']:.4f}"
        )

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------
    model.eval()

    ot_preds = []
    ot_trues = []

    off_preds = []
    off_trues = []

    with torch.no_grad():

        for x_seq, x_aux, y in get_dataloader(
            "data/processed/ontarget_splits.npz",
            "test",
            batch_size,
            shuffle=False,
        ):

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

        for x_pair, y in off_test:

            predictions = torch.sigmoid(
                model.forward_offtarget(x_pair)
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

    print(
        f"FINAL TEST — "
        f"OT Spearman: {ot_test_metrics['spearman']:.4f}, "
        f"OT Pearson: {ot_test_metrics['pearson']:.4f} | "
        f"OFF AUPRC: {off_test_metrics['auprc']:.4f}"
    )

    return ot_test_metrics, off_test_metrics


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Use the same architecture and seeds as our previous
    # diagnostic experiments.
    # ---------------------------------------------------------
    random.seed(14)

    arch = sample_architecture()

    print(
        "Sampled architecture:",
        [layer["block_type"] for layer in arch],
    )

    # ---------------------------------------------------------
    # Test one alpha at a time.
    #
    # Start with 0.80 only.
    # We will decide whether to continue to 0.90/0.95
    # after seeing the result.
    # ---------------------------------------------------------
    alpha = 0.80

    print(
        f"\nTesting alpha = {alpha:.2f}"
    )

    torch.manual_seed(42)

    model = MultiTaskNASModel(arch)

    train_multitask(
        model,
        epochs=5,
        batch_size=32,
        lr=1e-3,
        alpha=alpha,
    )
