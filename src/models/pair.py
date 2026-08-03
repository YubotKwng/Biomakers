"""Paired-encoding −SRM model.
"""
from __future__ import annotations

import torch.nn as nn

from .mlp import MLPEncoder


class PairModel(nn.Module):
    """Shared-encoder paired model: progression score from latent delta,
    plus FARS / SARA heads at each visit.

    The model compares baseline and follow-up feature vectors through a shared
    encoder and returns progression plus clinical-score heads.
    """

    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.enc = MLPEncoder(in_dim, hidden=8, out_dim=4, dropout=dropout)
        self.prog = nn.Linear(4, 1)
        self.fars = nn.Linear(4, 1)
        self.sara = nn.Linear(4, 1)

    def forward(self, x1, x2):
        z1 = self.enc(x1)
        z2 = self.enc(x2)
        dz = z2 - z1
        prog = self.prog(dz)
        fars1 = self.fars(z1)
        fars2 = self.fars(z2)
        sara1 = self.sara(z1)
        sara2 = self.sara(z2)
        return prog, fars1, fars2, sara1, sara2


__all__ = ["PairModel"]
