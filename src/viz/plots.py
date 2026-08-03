"""Plotting helpers for SHAP summaries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Default number of features to show in compact importance plots.
TOP_N_FEATURES = 15


def _plot_top_pos_neg(mean_shap_1d, feature_names, title, top_n=TOP_N_FEATURES, save_path=None):
    """Plot top-positive and top-negative mean SHAP features.

    When ``save_path`` is provided, the positive figure is saved to that path
    and the negative figure is saved beside it with a ``__neg`` suffix.
    """
    import matplotlib.pyplot as plt

    if len(feature_names) == 0:
        print(f"[skip] {title} (no features)")
        return

    vals = np.asarray(mean_shap_1d).reshape(-1)
    if len(vals) != len(feature_names):
        raise ValueError(
            f"Length mismatch for {title}: got {len(vals)} values vs {len(feature_names)} feature names. "
            "This usually means SHAP returned an unexpected tensor shape."
        )

    s = pd.Series(vals, index=feature_names)

    # Top positive
    s_pos = s.sort_values(ascending=False).head(top_n)
    if len(s_pos) > 0:
        s_pos = s_pos.iloc[::-1]
        h = max(3.5, 0.28 * len(s_pos) + 1.0)
        fig = plt.figure(figsize=(7.5, h))
        plt.barh(s_pos.index, s_pos.values)
        plt.title(f"{title} — Top +")
        plt.xlabel('mean(SHAP)')
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(str(save_path), dpi=150)
        plt.show()
        plt.close(fig)

    # Top negative
    s_neg = s.sort_values(ascending=True).head(top_n)
    if len(s_neg) > 0:
        s_neg = s_neg.iloc[::-1]
        h = max(3.5, 0.28 * len(s_neg) + 1.0)
        fig = plt.figure(figsize=(7.5, h))
        plt.barh(s_neg.index, s_neg.values)
        plt.title(f"{title} — Top −")
        plt.xlabel('mean(SHAP)')
        plt.tight_layout()
        if save_path is not None:
            p = Path(str(save_path))
            neg_path = p.with_name(p.stem + "__neg" + p.suffix)
            fig.savefig(str(neg_path), dpi=150)
        plt.show()
        plt.close(fig)


def _plot_modality_summary(modality_to_value, title, save_path=None):
    """Horizontal bar of summed mean(SHAP) per modality.

    Plot the largest positive and negative feature attributions and
    optionally save the figure to disk.
    """
    import matplotlib.pyplot as plt

    s = pd.Series(modality_to_value).sort_values(ascending=True)
    fig = plt.figure(figsize=(6.5, 3.2))
    plt.barh(s.index, s.values)
    plt.title(title)
    plt.xlabel('sum(mean(SHAP))')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
    plt.show()
    plt.close(fig)


__all__ = ["_plot_top_pos_neg", "_plot_modality_summary"]
