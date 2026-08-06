"""Multimodal fusion MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mlp import MLPEncoder


class ZeroEncoder(nn.Module):
    """Return a zero latent vector for absent modalities."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = int(out_dim)

    def forward(self, x):
        """Return a latent zero block matching the current batch size."""
        return x.new_zeros((x.shape[0], self.out_dim))


class FusionModel(nn.Module):
    """Three-branch fusion MLP (struct + diff + back) with FARS / SARA /
    progression heads.

    The model uses separate modality encoders before fusing them into clinical
    and progression heads.
    """

    def __init__(self, dims, dropout=0.3):
        super().__init__()
        self.enc_struct = (
            MLPEncoder(dims['struct'], hidden=8, out_dim=4, dropout=dropout)
            if int(dims['struct']) > 0 else ZeroEncoder(out_dim=4)
        )
        self.enc_diff = (
            MLPEncoder(dims['diff'], hidden=8, out_dim=4, dropout=dropout)
            if int(dims['diff']) > 0 else ZeroEncoder(out_dim=4)
        )
        self.enc_back = (
            MLPEncoder(dims['back'], hidden=4, out_dim=2, dropout=dropout)
            if int(dims['back']) > 0 else ZeroEncoder(out_dim=2)
        )

        fused_dim = 4 + 4 + 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 4), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fars = nn.Linear(4, 1)
        self.sara = nn.Linear(4, 1)
        self.prog = nn.Linear(4, 1)

    def forward(self, x_struct, x_diff, x_back):
        """Encode modalities, fuse them, and return FARS, SARA, progression."""
        h = torch.cat([
            self.enc_struct(x_struct),
            self.enc_diff(x_diff),
            self.enc_back(x_back)
        ], dim=1)
        h = self.head(h)
        fars = self.fars(h)
        sara = self.sara(h)
        prog = self.prog(h)
        return fars, sara, prog


__all__ = ["FusionModel", "ZeroEncoder"]
