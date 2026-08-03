"""Interaction-term feature construction (X⊗Z).

Interaction columns support patient-adaptive biomarker weighting by combining imaging features with clinical or genetic modulators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expand_interactions(X: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    """Build the X⊗Z interaction matrix as a DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Imaging features, shape (n, p). Numeric.
    Z : pd.DataFrame
        Modulator features, shape (n, q). Numeric.

    Returns
    -------
    pd.DataFrame
        Shape (n, p*q). Column names are ``f"{x_col} × {z_col}"`` ordered
        by (x_col outer, z_col inner) to match the
        ``np.einsum('ij,ik->ijk', X, Z).reshape(n, -1)`` layout used in the
        downstream linear composite. Index is preserved from ``X``.

    Raises
    ------
    ValueError
        If ``X`` and ``Z`` have a different number of rows.
    """
    if len(X) != len(Z):
        raise ValueError(f"X and Z row count mismatch: {len(X)} vs {len(Z)}")
    n = len(X)
    inter = np.einsum("ij,ik->ijk", X.values, Z.values).reshape(n, -1)
    cols = [f"{xc} × {zc}" for xc in X.columns for zc in Z.columns]
    return pd.DataFrame(inter, columns=cols, index=X.index)


__all__ = ["expand_interactions"]
