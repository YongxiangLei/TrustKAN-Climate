"""Strong, compact neural baselines using a common interface."""
from __future__ import annotations

import math
import torch
from torch import nn


class MLPForecaster(nn.Module):
    def __init__(self, history: int, n_features: int, horizon: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history * n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, horizon),
        )
    def forward(self, x): return self.net(x)


class RNNForecaster(nn.Module):
    def __init__(self, kind: str, n_features: int, horizon: int, hidden=64, layers=2):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind.lower()]
        self.rnn = cls(n_features, hidden, layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(z[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, d_model=64, heads=4, layers=2):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, 4*d_model, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(d_model, horizon)
    def forward(self, x):
        z = self.encoder(self.pos(self.input(x)))
        return self.head(z[:, -1])


class PersistenceForecaster:
    """Deterministic last-observation baseline."""
    def __init__(self, horizon: int): self.horizon = horizon
    def predict(self, x):
        x = torch.as_tensor(x)
        return x[:, -1, 0:1].repeat(1, self.horizon)
