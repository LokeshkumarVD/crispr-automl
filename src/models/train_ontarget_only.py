import random

import numpy as np
import torch
import torch.nn as nn

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics
from src.nas.search_space import sample_architecture


def train_ontarget_only(model, epochs=5, batch_size=32, lr=1e-3):
    """
    Train the same NAS model using ONLY the on-target regression task.

    This is a diagnostic experiment:
    - Same architecture sampling
    - Same on-target data
    - Same optimizer
    - Same learning rate
    - Same number of epochs
    - No off-target task
    """

    # ---------------------------------------------------------
    # Load on-target data
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

    ot_test = get_dataloader(
        "data/processed/ontarget_splits.npz",
        "test",
        batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Loss and optimizer
    # ---------------------------------------------------------
    mse_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        n_steps = 0

        for x_seq, x_aux, y_ot in ot_train:
            optimizer.zero_grad()

            # On-target prediction only
            pred_ot = model.forward_ontarget(x_seq, x_aux)

            # Regression loss
            loss_ot = mse_loss(pred_ot, y_ot)

            # Backpropagation
            loss_ot.backward()

            optimizer.step()

            total_loss += loss_ot.item()
            n_steps += 1

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        model.eval()

        ot_preds = []
        ot_trues = []

        with torch.no_grad():
            for x_seq, x_aux, y in ot_val:
                predictions = model.forward_ontarget(
                    x_seq,
                    x_aux,
                )

                ot_preds.extend(predictions.numpy())
                ot_trues.extend(y.numpy())

        ot_metrics = regression_metrics(
            ot_trues,
            ot_preds,
        )

        avg_train_loss = total_loss / n_steps

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"Train OT loss: {avg_train_loss:.4f} | "
            f"Val Spearman: {ot_metrics['spearman']:.4f} | "
            f"Val Pearson: {ot_metrics['pearson']:.4f}"
        )

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------
    model.eval()

    ot_preds = []
    ot_trues = []

    with torch.no_grad():
        for x_seq, x_aux, y in ot_test:
            predictions = model.forward_ontarget(
                x_seq,
                x_aux,
            )

            ot_preds.extend(predictions.numpy())
            ot_trues.extend(y.numpy())

    ot_test_metrics = regression_metrics(
        ot_trues,
        ot_preds,
    )

    print(
        f"\nFINAL TEST — "
        f"On-target Spearman: {ot_test_metrics['spearman']:.4f}, "
        f"Pearson: {ot_test_metrics['pearson']:.4f}"
    )

    return ot_test_metrics


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Keep the same architecture sampling setup as the
    # multitask diagnostic experiment.
    # ---------------------------------------------------------
    random.seed(14)

    arch = sample_architecture()

    print(
        "Sampled architecture:",
        [layer["block_type"] for layer in arch],
    )

    # Keep the same PyTorch seed used in the multitask experiment.
    torch.manual_seed(42)

    model = MultiTaskNASModel(arch)

    train_ontarget_only(
        model,
        epochs=5,
        batch_size=32,
        lr=1e-3,
    )
