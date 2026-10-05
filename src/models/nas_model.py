# src/models/nas_model.py
import torch
import torch.nn as nn
from src.models.blocks import CNNBlock, BiLSTMBlock, AttentionBlock


class CNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dropout):
        super().__init__()
        self.block = CNNBlock(in_dim, out_dim, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):  # x: (batch, seq_len, dim)
        x = self.block(x.permute(0, 2, 1))  # CNN wants (batch, dim, seq_len)
        return x.permute(0, 2, 1)


class BiLSTMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.block = BiLSTMBlock(in_dim, out_dim // 2)  # bidirectional doubles the size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.block(x))


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)  # keeps out_dim divisible by num_heads
        self.block = AttentionBlock(out_dim, num_heads=num_heads, dropout=dropout)
        self.last_attn = None  # saved for Phase 6 explainability

    def forward(self, x):
        x = self.proj(x)
        x, weights = self.block(x)
        self.last_attn = weights.detach()
        return x


class NASModel(nn.Module):
    """Builds a network from a list of layer configs (see search_space.py).
    Input: (batch, seq_len, in_channels). For on-target: in_channels=4, aux_dim=3.
    For off-target: in_channels=8, aux_dim=0 (permute data to (batch, 23, 8) first)."""

    def __init__(self, architecture, in_channels=4, aux_dim=3):
        super().__init__()
        layers, dim = [], in_channels
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
        self.aux_dim = aux_dim
        self.head = nn.Sequential(nn.Linear(dim + aux_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_seq, x_aux=None):
        x = self.backbone(x_seq).mean(dim=1)  # pool over sequence positions
        if self.aux_dim > 0:
            x = torch.cat([x, x_aux], dim=1)
        return self.head(x).squeeze(-1)


if __name__ == "__main__":
    import random
    from src.nas.search_space import sample_architecture

    random.seed(0)
    x_seq, x_aux = torch.randn(8, 23, 4), torch.randn(8, 3)
    for i in range(5):
        arch = sample_architecture()
        model = NASModel(arch)
        out = model(x_seq, x_aux)
        n = sum(p.numel() for p in model.parameters())
        print(f"Candidate {i+1}: {[l['block_type'] for l in arch]} -> output {tuple(out.shape)}, params {n:,}")

    model = NASModel(sample_architecture(), in_channels=8, aux_dim=0)
    print("Off-target style output shape:", tuple(model(torch.randn(8, 23, 8)).shape))