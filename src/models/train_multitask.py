# src/models/train_multitask.py
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics


ALPHA = 0.7  # weight on on-target loss; (1 - ALPHA) on off-target loss

ONTARGET_NPZ = "data/processed/ontarget_splits.npz"
OFFTARGET_NPZ = "data/processed/offtarget_crispr_splits.npz"


def get_offtarget_loaders(batch_size=32):
    data = np.load(OFFTARGET_NPZ)

    def make(split, shuffle):
        ds = TensorDataset(
            torch.tensor(data[f"X_{split}"], dtype=torch.float32),
            torch.tensor(data[f"y_{split}"], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    pos_weight = torch.tensor(
        [(data["y_train"] == 0).sum() / max((data["y_train"] == 1).sum(), 1)],
        dtype=torch.float32,
    )

    return make("train", True), make("val", False), make("test", False), pos_weight


def gradient_norm(model):
    """L2 norm of all currently stored gradients (used only when diagnostics=True)."""
    total_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total_sq += parameter.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def evaluate_split(model, split, batch_size=32):
    """Evaluate a model on 'val' or 'test' for BOTH tasks.
    Returns (ontarget_metrics, offtarget_metrics).
    Calling this with split='test' should be done once, on the final chosen model."""
    assert split in ("val", "test"), "split must be 'val' or 'test'"

    ot_loader = get_dataloader(ONTARGET_NPZ, split, batch_size, shuffle=False)
    _, off_val, off_test, _ = get_offtarget_loaders(batch_size)
    off_loader = off_val if split == "val" else off_test

    model.eval()
    ot_preds, ot_trues = [], []
    off_preds, off_trues = [], []

    with torch.no_grad():
        for x_seq, x_aux, y in ot_loader:
            ot_preds.extend(model.forward_ontarget(x_seq, x_aux).numpy())
            ot_trues.extend(y.numpy())

        for x_pair, y in off_loader:
            off_preds.extend(torch.sigmoid(model.forward_offtarget(x_pair)).numpy())
            off_trues.extend(y.numpy())

    return (
        regression_metrics(ot_trues, ot_preds),
        classification_metrics(off_trues, off_preds),
    )


def train_multitask(
    model,
    epochs=5,
    batch_size=32,
    lr=1e-3,
    alpha=ALPHA,
    diagnostics=False,
    verbose=True,
):
    """Jointly train both tasks through the shared backbone.

    Returns a dict with the FINAL-EPOCH VALIDATION metrics:
        {"val_ot": {...spearman, pearson...}, "val_off": {...auprc...}}

    This function never touches the test set. Use evaluate_split(model, "test")
    once, after the final model has been chosen using validation scores.

    diagnostics=True re-enables the per-task gradient-norm measurements
    (extra backward passes, noticeably slower). They do not change the weight update.
    """
    ot_train = get_dataloader(ONTARGET_NPZ, "train", batch_size, shuffle=True)
    off_train, _, _, pos_weight = get_offtarget_loaders(batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Running average of each task's loss magnitude, used to put both losses
    # on a comparable scale before combining them.
    running_ot = 1.0
    running_off = 1.0
    momentum = 0.9

    val_ot, val_off = None, None

    for epoch in range(epochs):
        model.train()

        total_ot_loss = 0.0
        total_off_loss = 0.0
        n_steps = 0
        ot_gradients, off_gradients = [], []

        # NOTE: itertools.cycle replays the same off-target batches (same order)
        # when it restarts within an epoch. Kept unchanged so results stay
        # comparable with earlier experiments.
        off_cycle = itertools.cycle(off_train)

        for (x_seq, x_aux, y_ot), (x_pair, y_off) in zip(ot_train, off_cycle):
            optimizer.zero_grad()

            pred_ot = model.forward_ontarget(x_seq, x_aux)
            pred_off = model.forward_offtarget(x_pair)

            loss_ot = mse_loss(pred_ot, y_ot)
            loss_off = bce_loss(pred_off, y_off)

            running_ot = momentum * running_ot + (1 - momentum) * loss_ot.item()
            running_off = momentum * running_off + (1 - momentum) * loss_off.item()

            norm_loss_ot = loss_ot / (running_ot + 1e-8)
            norm_loss_off = loss_off / (running_off + 1e-8)

            if diagnostics:
                optimizer.zero_grad()
                norm_loss_ot.backward(retain_graph=True)
                ot_gradients.append(gradient_norm(model))

                optimizer.zero_grad()
                norm_loss_off.backward(retain_graph=True)
                off_gradients.append(gradient_norm(model))

                optimizer.zero_grad()

            loss = alpha * norm_loss_ot + (1 - alpha) * norm_loss_off
            loss.backward()
            optimizer.step()

            total_ot_loss += loss_ot.item()
            total_off_loss += loss_off.item()
            n_steps += 1

        # Validation only (never test) after every epoch
        val_ot, val_off = evaluate_split(model, "val", batch_size)

        if verbose:
            line = (
                f"  Epoch {epoch + 1}/{epochs} | "
                f"OT loss {total_ot_loss / n_steps:.4f} | "
                f"OFF loss {total_off_loss / n_steps:.4f} | "
                f"val Spearman {val_ot['spearman']:.4f} | "
                f"val AUPRC {val_off['auprc']:.4f}"
            )
            if diagnostics:
                mean_ot = float(np.mean(ot_gradients))
                mean_off = float(np.mean(off_gradients))
                line += f" | grad OT {mean_ot:.3f} OFF {mean_off:.3f} ratio {mean_off / (mean_ot + 1e-8):.2f}x"
            print(line)

    return {"val_ot": val_ot, "val_off": val_off}