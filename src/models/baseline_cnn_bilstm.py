# src/models/baseline_cnn_bilstm.py
import torch
import torch.nn as nn
from src.models.blocks import CNNBlock, BiLSTMBlock


class CNNBiLSTMBaseline(nn.Module):
    """Baseline model: CNN -> BiLSTM -> pooled -> two separate output heads
    (on-target regression + off-target classification)."""

    def __init__(self, cnn_out_channels=16, lstm_hidden=32):
        super().__init__()
        self.cnn = CNNBlock(in_channels=4, out_channels=cnn_out_channels)
        self.bilstm = BiLSTMBlock(input_size=cnn_out_channels, hidden_size=lstm_hidden)

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # On-target head: takes pooled sequence features + 3 auxiliary features
        self.ontarget_head = nn.Sequential(
            nn.Linear(lstm_out_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Off-target head: same backbone, but off-target data has no auxiliary features
        # (it's already an 8-channel encoded pair), so this head is trained separately
        self.offtarget_cnn = CNNBlock(in_channels=8, out_channels=cnn_out_channels)
        self.offtarget_bilstm = BiLSTMBlock(input_size=cnn_out_channels, hidden_size=lstm_hidden)
        self.offtarget_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward_ontarget(self, x_seq, x_aux):
        # x_seq: (batch, 23, 4) -> permute for CNN -> (batch, 4, 23)
        x = self.cnn(x_seq.permute(0, 2, 1))
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, channels) for BiLSTM
        x = self.bilstm(x)
        x = x.mean(dim=1)  # pool across sequence positions -> (batch, lstm_out_dim)
        x = torch.cat([x, x_aux], dim=1)  # concatenate auxiliary features
        return self.ontarget_head(x).squeeze(-1)

    def forward_offtarget(self, x_pair):
        # x_pair: (batch, 8, 23) -> already channel-first, matches encoding format
        x = self.offtarget_cnn(x_pair)
        x = x.permute(0, 2, 1)
        x = self.offtarget_bilstm(x)
        x = x.mean(dim=1)
        return self.offtarget_head(x).squeeze(-1)


if __name__ == "__main__":
    model = CNNBiLSTMBaseline()

    # Test on-target path with real-shaped dummy data
    x_seq = torch.randn(8, 23, 4)
    x_aux = torch.randn(8, 3)
    ontarget_out = model.forward_ontarget(x_seq, x_aux)
    print(f"On-target output shape: {ontarget_out.shape}")  # expect (8,)

    # Test off-target path
    x_pair = torch.randn(8, 8, 23)
    offtarget_out = model.forward_offtarget(x_pair)
    print(f"Off-target output shape: {offtarget_out.shape}")  # expect (8,)