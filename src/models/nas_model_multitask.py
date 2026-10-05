# src/models/nas_model_multitask.py
import torch
import torch.nn as nn
from src.models.nas_model import CNNLayer, BiLSTMLayer, AttentionLayer


class MultiTaskNASModel(nn.Module):
    """One shared backbone (built from a sampled architecture encoding),
    with two separate output heads: on-target regression + off-target classification.

    Shared input format: (batch, seq_len=23, channels=8).
    - On-target data is (batch, 23, 4) -> zero-padded to 8 channels before entering the backbone.
    - Off-target data is already (batch, 8, 23) raw -> permuted to (batch, 23, 8).
    """

    def __init__(self, architecture, aux_dim=3, shared_channels=8):
        super().__init__()
        layers, dim = [], shared_channels
        for cfg in architecture:
            t, out, dr = cfg["block_type"], cfg["channels"], cfg["dropout"]
            if t == "cnn_small":
                layers.append(CNNLayer(dim, out, 3, dr))
            elif t == "cnn_large":
                layers.append(CNNLayer(dim, out, 5, dr))
            elif t == "bilstm":
                layers.append(BiLSTMLayer(dim, out, dr))
            elif t == "attention":
                layers.append(AttentionLayer(dim, out, cfg["num_heads"], dr))
            else:
                raise ValueError(f"Unknown block type: {t}")
            dim = out
        self.backbone = nn.Sequential(*layers)

        self.ontarget_head = nn.Sequential(
            nn.Linear(dim + aux_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.offtarget_head = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def _pad_ontarget(self, x_seq):
        """(batch, 23, 4) -> (batch, 23, 8), zero-padded in the extra 4 channels."""
        batch, seq_len, _ = x_seq.shape
        pad = torch.zeros(batch, seq_len, 4, device=x_seq.device, dtype=x_seq.dtype)
        return torch.cat([x_seq, pad], dim=2)

    def forward_ontarget(self, x_seq, x_aux):
        x = self._pad_ontarget(x_seq)
        feat = self.backbone(x).mean(dim=1)          # pool across sequence positions
        feat = torch.cat([feat, x_aux], dim=1)
        return self.ontarget_head(feat).squeeze(-1)

    def forward_offtarget(self, x_pair):
        # raw off-target data is (batch, 8, 23) -> permute to (batch, 23, 8)
        x = x_pair.permute(0, 2, 1)
        feat = self.backbone(x).mean(dim=1)
        return self.offtarget_head(feat).squeeze(-1)


if __name__ == "__main__":
    from src.nas.search_space import sample_architecture

    arch = sample_architecture()
    print("Sampled shared-backbone architecture:")
    for i, l in enumerate(arch):
        print(f"  Layer {i+1}: {l}")

    model = MultiTaskNASModel(arch)

    # On-target dummy input, matching real shapes
    x_seq = torch.randn(8, 23, 4)
    x_aux = torch.randn(8, 3)
    ontarget_out = model.forward_ontarget(x_seq, x_aux)
    print(f"\nOn-target output shape: {ontarget_out.shape}")  # expect (8,)

    # Off-target dummy input, matching real raw shape (N, 8, 23)
    x_pair = torch.randn(8, 8, 23)
    offtarget_out = model.forward_offtarget(x_pair)
    print(f"Off-target output shape: {offtarget_out.shape}")  # expect (8,)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal shared model parameters: {n_params:,}")