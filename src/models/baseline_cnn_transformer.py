# src/models/baseline_cnn_transformer.py
import torch
import torch.nn as nn
from src.models.blocks import CNNBlock, AttentionBlock


class CNNTransformerBaseline(nn.Module):
    """Baseline model: CNN -> Attention -> pooled -> two separate output heads
    (on-target regression + off-target classification)."""

    def __init__(self, cnn_out_channels=16, num_heads=4):
        super().__init__()
        self.cnn = CNNBlock(in_channels=4, out_channels=cnn_out_channels)
        self.attention = AttentionBlock(embed_dim=cnn_out_channels, num_heads=num_heads)

        self.ontarget_head = nn.Sequential(
            nn.Linear(cnn_out_channels + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.offtarget_cnn = CNNBlock(in_channels=8, out_channels=cnn_out_channels)
        self.offtarget_attention = AttentionBlock(embed_dim=cnn_out_channels, num_heads=num_heads)
        self.offtarget_head = nn.Sequential(
            nn.Linear(cnn_out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward_ontarget(self, x_seq, x_aux):
        x = self.cnn(x_seq.permute(0, 2, 1))       # (batch, channels, seq_len)
        x = x.permute(0, 2, 1)                      # (batch, seq_len, channels) for attention
        x, _ = self.attention(x)
        x = x.mean(dim=1)                           # pool across sequence positions
        x = torch.cat([x, x_aux], dim=1)
        return self.ontarget_head(x).squeeze(-1)

    def forward_offtarget(self, x_pair):
        x = self.offtarget_cnn(x_pair)
        x = x.permute(0, 2, 1)
        x, _ = self.offtarget_attention(x)
        x = x.mean(dim=1)
        return self.offtarget_head(x).squeeze(-1)


if __name__ == "__main__":
    model = CNNTransformerBaseline()

    x_seq = torch.randn(8, 23, 4)
    x_aux = torch.randn(8, 3)
    ontarget_out = model.forward_ontarget(x_seq, x_aux)
    print(f"On-target output shape: {ontarget_out.shape}")  # expect (8,)

    x_pair = torch.randn(8, 8, 23)
    offtarget_out = model.forward_offtarget(x_pair)
    print(f"Off-target output shape: {offtarget_out.shape}")  # expect (8,)