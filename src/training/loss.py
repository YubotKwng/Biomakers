"""Differentiable paired-progression loss.
"""
from __future__ import annotations

import numpy as np
import torch


def paired_progression_loss(scores, visits, subjects):
    """Differentiable paired progression loss.

    scores: torch.Tensor shape (n,) or (n,)
    visits: numpy array length n with values 1/2
    subjects: numpy array length n with subject ids

    Returns negative SRM-like objective: -mean(delta) / std(delta)
    where delta = score(V2) - score(V1) for each subject.

    Maximizes standardized follow-up-minus-baseline progression change.
    """
    deltas = []
    # Python loop is fine for n~50; keeps computation in torch graph.
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        if idx.size != 2:
            continue
        v = visits[idx]
        # Identify v1 and v2 positions
        try:
            i1 = idx[np.where(v == 1)[0][0]]
            i2 = idx[np.where(v == 2)[0][0]]
        except Exception:
            continue
        deltas.append(scores[i2] - scores[i1])

    if len(deltas) < 2:
        # Return 0 loss but keep it connected to graph if possible
        return scores.sum() * 0.0

    d = torch.stack(deltas)
    return -(d.mean() / (d.std(unbiased=True) + 1e-6))


def neg_srm_loss(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return ``-mean(delta) / (std(delta, unbiased=True) + eps)``.

    The caller supplies already-paired per-subject ``v2 - v1`` differences.
    This is used by interaction models when the pairing index is computed
    once at fit time.

    Parameters
    ----------
    delta : torch.Tensor
        1-D tensor of within-subject score differences (visit2 − visit1),
        one entry per subject.
    eps : float, default 1e-6
        Numerical floor added to the SD denominator.

    Returns
    -------
    torch.Tensor
        Scalar; negative of the paired Cohen's d_z. Minimise to maximise
        the SRM.
    """
    return -delta.mean() / (delta.std(unbiased=True) + eps)


__all__ = ["paired_progression_loss", "neg_srm_loss"]
