from __future__ import annotations
import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
