# src/models/train_multitask.py
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.nas_model_multitask import MultiTaskNASModel
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics

ALPHA = 0.7  # weight on on-target loss; (1-ALPHA) on off-target loss


def get_offtarget_loaders(batch_size=32):
    data = np.load("data/processed/offtarget_crispr_splits.npz")

    def make(split, shuffle):
        ds = TensorDataset(
            torch.tensor(data[f"X_{split}"], dtype=torch.float32),
            torch.tensor(data[f"y_{split}"], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    pos_weight = torch.tensor(
        [(data["y_train"] == 0).sum() / max((data["y_train"] == 1).sum(), 1)]
    )
    return make("train", True), make("val", False), make("test", False), pos_weight


def train_multitask(model, epochs=5, batch_size=32, lr=1e-3, alpha=ALPHA):
    ot_train = get_dataloader("data/processed/ontarget_splits.npz", "train", batch_size, shuffle=True)
    ot_val = get_dataloader("data/processed/ontarget_splits.npz", "val", batch_size, shuffle=False)
    off_train, off_val, off_test, pos_weight = get_offtarget_loaders(batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Running average of each task's raw loss magnitude, used to normalize them
    # onto a comparable scale before combining — prevents one task's larger loss
    # values from dominating shared-backbone gradients regardless of alpha.
    running_ot, running_off = 1.0, 1.0
    momentum = 0.9

    for epoch in range(epochs):
        model.train()
        total_ot_loss, total_off_loss, n_steps = 0.0, 0.0, 0

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

            loss = alpha * norm_loss_ot + (1 - alpha) * norm_loss_off

            loss.backward()
            optimizer.step()

            total_ot_loss += loss_ot.item()
            total_off_loss += loss_off.item()
            n_steps += 1

        model.eval()
        ot_preds, ot_trues = [], []
        off_preds, off_trues = [], []
        with torch.no_grad():
            for x_seq, x_aux, y in ot_val:
                ot_preds.extend(model.forward_ontarget(x_seq, x_aux).numpy())
                ot_trues.extend(y.numpy())
            for x_pair, y in off_val:
                off_preds.extend(torch.sigmoid(model.forward_offtarget(x_pair)).numpy())
                off_trues.extend(y.numpy())

        ot_metrics = regression_metrics(ot_trues, ot_preds)
        off_metrics = classification_metrics(off_trues, off_preds)

        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"OnT loss: {total_ot_loss/n_steps:.4f}, OffT loss: {total_off_loss/n_steps:.4f} | "
            f"val Spearman: {ot_metrics['spearman']:.4f}, val AUPRC: {off_metrics['auprc']:.4f}"
        )

    model.eval()
    ot_test_loader = get_dataloader("data/processed/ontarget_splits.npz", "test", batch_size, shuffle=False)
    ot_preds, ot_trues = [], []
    off_preds, off_trues = [], []
    with torch.no_grad():
        for x_seq, x_aux, y in ot_test_loader:
            ot_preds.extend(model.forward_ontarget(x_seq, x_aux).numpy())
            ot_trues.extend(y.numpy())
        for x_pair, y in off_test:
            off_preds.extend(torch.sigmoid(model.forward_offtarget(x_pair)).numpy())
            off_trues.extend(y.numpy())

    ot_test_metrics = regression_metrics(ot_trues, ot_preds)
    off_test_metrics = classification_metrics(off_trues, off_preds)
    print(f"\nFINAL TEST — On-target Spearman: {ot_test_metrics['spearman']:.4f}, "
          f"Pearson: {ot_test_metrics['pearson']:.4f} | Off-target AUPRC: {off_test_metrics['auprc']:.4f}")

    return ot_test_metrics, off_test_metrics


if __name__ == "__main__":
    import random
    from src.nas.search_space import sample_architecture

    random.seed(14)   # ['cnn_small', 'bilstm', 'attention'] — tests attention under multi-task training
    arch = sample_architecture()
    print("Sampled architecture:", [l["block_type"] for l in arch])

    torch.manual_seed(42)
    model = MultiTaskNASModel(arch)
    train_multitask(model, epochs=5, alpha=ALPHA)