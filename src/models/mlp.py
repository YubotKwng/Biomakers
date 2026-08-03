"""MLP encoder used by FusionModel and PairModel.
"""
from __future__ import annotations

import torch.nn as nn


class MLPEncoder(nn.Module):
    """Two-layer MLP with ReLU + Dropout used as a per-modality encoder.

    The model maps standardized feature vectors to a scalar progression score.
    """

    def __init__(self, in_dim, hidden=8, out_dim=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


__all__ = ["MLPEncoder"]
