# src/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class OnTargetDataset(Dataset):
    """Wraps the on-target sequence + auxiliary feature + efficiency label data
    for a given split (train/val/test)."""

    def __init__(self, npz_path, split="train"):
        data = np.load(npz_path)
        self.X_seq = torch.tensor(data[f"X_seq_{split}"], dtype=torch.float32)
        self.X_aux = torch.tensor(data[f"X_aux_{split}"], dtype=torch.float32)
        self.y = torch.tensor(data[f"y_{split}"], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_aux[idx], self.y[idx]


def get_dataloader(npz_path, split="train", batch_size=32, shuffle=True):
    dataset = OnTargetDataset(npz_path, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    npz_path = "data/processed/ontarget_splits.npz"

    for split in ["train", "val", "test"]:
        loader = get_dataloader(npz_path, split=split, batch_size=8, shuffle=(split == "train"))
        X_seq_batch, X_aux_batch, y_batch = next(iter(loader))
        print(f"{split} — X_seq: {X_seq_batch.shape}, X_aux: {X_aux_batch.shape}, y: {y_batch.shape}")