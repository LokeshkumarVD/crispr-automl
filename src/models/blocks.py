# src/models/blocks.py
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """1D convolution over the sequence, treating each of the 4 (or 8) channels
    as input features at each of the 23 sequence positions."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x expected shape: (batch, channels, seq_len)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM over the sequence."""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x expected shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return out  # shape: (batch, seq_len, hidden_size * 2)


class AttentionBlock(nn.Module):
    """Standard multi-head self-attention block."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x expected shape: (batch, seq_len, embed_dim)
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)  # residual connection
        return x, attn_weights  # weights returned now — useful later for Phase 6 explainability


if __name__ == "__main__":
    # Quick sanity check with dummy data matching your real batch shapes
    batch_size = 8

    # CNN expects (batch, channels, seq_len) — your one-hot data is (batch, seq_len, channels),
    # so you'll permute before feeding it in
    dummy_onehot = torch.randn(batch_size, 23, 4)  # matches your real X_seq shape
    cnn = CNNBlock(in_channels=4, out_channels=16)
    cnn_out = cnn(dummy_onehot.permute(0, 2, 1))  # -> (batch, 4, 23)
    print(f"CNN output shape: {cnn_out.shape}")  # expect (8, 16, 23)

    # BiLSTM expects (batch, seq_len, features)
    bilstm_input = cnn_out.permute(0, 2, 1)  # back to (batch, seq_len, channels)
    bilstm = BiLSTMBlock(input_size=16, hidden_size=32)
    bilstm_out = bilstm(bilstm_input)
    print(f"BiLSTM output shape: {bilstm_out.shape}")  # expect (8, 23, 64) — 64 = 32*2 (bidirectional)

    # Attention expects (batch, seq_len, embed_dim)
    attn = AttentionBlock(embed_dim=64, num_heads=4)
    attn_out, attn_weights = attn(bilstm_out)
    print(f"Attention output shape: {attn_out.shape}")  # expect (8, 23, 64)
    print(f"Attention weights shape: {attn_weights.shape}")  # expect (8, 23, 23)