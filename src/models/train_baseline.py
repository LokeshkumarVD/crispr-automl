# src/models/train_baseline.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.baseline_cnn_bilstm import CNNBiLSTMBaseline
from src.data.dataset import get_dataloader
from src.eval.metrics import regression_metrics, classification_metrics

DEVICE = torch.device("cpu")  # no GPU available on this laptop


def train_ontarget(model, epochs=5, batch_size=32, lr=1e-3):
    npz_path = "data/processed/ontarget_splits.npz"
    train_loader = get_dataloader(npz_path, split="train", batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(npz_path, split="val", batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_seq, x_aux, y in train_loader:
            optimizer.zero_grad()
            pred = model.forward_ontarget(x_seq, x_aux)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x_seq, x_aux, y in val_loader:
                pred = model.forward_ontarget(x_seq, x_aux)
                all_preds.extend(pred.numpy())
                all_true.extend(y.numpy())

        metrics = regression_metrics(all_true, all_preds)
        print(f"[On-target] Epoch {epoch+1}/{epochs} — train loss: {total_loss/len(train_loader):.4f} "
              f"— val Spearman: {metrics['spearman']:.4f}, Pearson: {metrics['pearson']:.4f}")


def train_offtarget(model, epochs=5, batch_size=32, lr=1e-3):
    data = np.load("data/processed/offtarget_crispr.npz")
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    dataset = TensorDataset(X, y)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Weighted loss to counter the severe 0.55% positive imbalance
    pos_weight = torch.tensor([(y == 0).sum().item() / max((y == 1).sum().item(), 1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model.forward_offtarget(x)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y_batch in val_loader:
                pred = torch.sigmoid(model.forward_offtarget(x))
                all_preds.extend(pred.numpy())
                all_true.extend(y_batch.numpy())

        metrics = classification_metrics(all_true, all_preds)
        print(f"[Off-target] Epoch {epoch+1}/{epochs} — train loss: {total_loss/len(train_loader):.4f} "
              f"— val AUPRC: {metrics['auprc']:.4f}")


if __name__ == "__main__":
    from src.models.baseline_cnn_transformer import CNNTransformerBaseline

    print("=== Training on-target (regression) — CNN-BiLSTM ===")
    model_ontarget = CNNBiLSTMBaseline()
    train_ontarget(model_ontarget, epochs=5)

    print("\n=== Training off-target (classification) — CNN-BiLSTM ===")
    model_offtarget = CNNBiLSTMBaseline()
    train_offtarget(model_offtarget, epochs=5)

    print("\n=== Training on-target (regression) — CNN-Transformer ===")
    model_ontarget_2 = CNNTransformerBaseline()
    train_ontarget(model_ontarget_2, epochs=5)

    print("\n=== Training off-target (classification) — CNN-Transformer ===")
    model_offtarget_2 = CNNTransformerBaseline()
    train_offtarget(model_offtarget_2, epochs=5)