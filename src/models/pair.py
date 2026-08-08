"""Paired-encoding −SRM model.
"""
from __future__ import annotations

import torch.nn as nn

from .mlp import MLPEncoder


class PairModel(nn.Module):
    """Shared-encoder paired model with an antisymmetric progression score.

    The model encodes both visits with shared weights, predicts one scalar
    progression score per visit, and returns ``score_visit2 - score_visit1``.
    This prevents the paired progression output from learning a constant bias,
    which would artificially inflate mean(delta) / std(delta).
    """

    def __init__(self, in_dim, dropout=0.3):
        super().__init__()
        self.enc = MLPEncoder(in_dim, hidden=8, out_dim=4, dropout=dropout)
        self.score = nn.Linear(4, 1)
        self.fars = nn.Linear(4, 1)
        self.sara = nn.Linear(4, 1)

    def forward(self, x1, x2):
        """Return progression as visit-2 score minus visit-1 score."""
        z1 = self.enc(x1)
        z2 = self.enc(x2)
        prog = self.score(z2) - self.score(z1)
        fars1 = self.fars(z1)
        fars2 = self.fars(z2)
        sara1 = self.sara(z1)
        sara2 = self.sara(z2)
        return prog, fars1, fars2, sara1, sara2


__all__ = ["PairModel"]
