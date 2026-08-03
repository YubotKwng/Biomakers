"""SHAP (Shapley) feature-importance helpers for FusionModel and PairModel.

Notes
-----
* DeepExplainer additivity checks fail intermittently for some torch
  graphs; ``SHAP_CHECK_ADDITIVITY = False`` is preserved as the default.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# Defaults for compact and reproducible SHAP summaries.
SHAP_BACKGROUND_SIZE = 40
SHAP_EVAL_SIZE = 120
TOP_N_FEATURES = 15
SHAP_CHECK_ADDITIVITY = False


# ---------------------------------------------------------------------------
# Tensor and array coercion helpers.
# ---------------------------------------------------------------------------
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_last_if_unit(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 2 and arr.shape[-1] == 1:
        return arr[:, 0]
    return arr


def _mean_shap_per_feature(shap_arr):
    v = _squeeze_last_if_unit(_as_numpy(shap_arr))
    if v.ndim != 2:
        raise ValueError(
            f"Expected SHAP values with shape (n_samples, n_features) or (n_samples, n_features, 1), got {v.shape}"
        )
    return v.mean(axis=0)


def _normalize_deep_shap_values(shap_values):
    # DeepExplainer can return:
    # - list[input] (single output)
    # - list[output][input] (multi-output)
    if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], list):
        shap_values = shap_values[0]
    return shap_values


def _shap_values_safe(explainer, inputs):
    """Call explainer.shap_values with additivity check control (SHAP version compatible)."""
    try:
        return explainer.shap_values(inputs, check_additivity=SHAP_CHECK_ADDITIVITY)
    except TypeError:
        # Some SHAP versions do not accept check_additivity.
        return explainer.shap_values(inputs)


# ---------------------------------------------------------------------------
# Sampling and combination-selection helpers.
# ---------------------------------------------------------------------------
def _sample_idx(n, k, rng):
    k = min(int(k), int(n))
    if k <= 0:
        return np.array([], dtype=int)
    if k == n:
        return np.arange(n)
    return rng.choice(np.arange(n), size=k, replace=False)


def _has_all_modalities(meta):
    return (
        len(meta.get('back_idx', [])) > 0
        and len(meta.get('struct_idx', [])) > 0
        and len(meta.get('diff_idx', [])) > 0
    )


def _pick_best_combo(results_df, combo_meta, prefer_all_modalities=True):
    df = results_df.sort_values('d', ascending=False).copy()
    if prefer_all_modalities:
        df_full = df[df['combination'].map(lambda name: _has_all_modalities(combo_meta[name]))]
        if len(df_full) > 0:
            return df_full.iloc[0]['combination'], True
    return df.iloc[0]['combination'], False


# ---------------------------------------------------------------------------
# SHAP orchestration.
# ---------------------------------------------------------------------------
def run_shap_on_combo(
    *,
    model_kind: str,
    combo_name: str,
    combinations: List[dict],
    combo_meta: Dict[str, dict],
    long_df: pd.DataFrame,
    subject_col: str,
    device: torch.device,
    seed: int = 42,
    background_size: int = SHAP_BACKGROUND_SIZE,
    eval_size: int = SHAP_EVAL_SIZE,
    train_kwargs: dict = None,
) -> dict:
    """Run DeepExplainer on a single combo for either ``"fusion"`` or
    ``"pair"`` and return per-modality mean-SHAP arrays + names.

    Trains the selected model, computes progression-head SHAP values, and
    aggregates feature attributions by modality.
    """
    import shap as _shap  # local import — heavy

    rng = np.random.RandomState(seed)

    if model_kind == "fusion":
        from ..training.fusion import _train_fusion_for_combo, _FusionProgWrapper

        model, arrays, feats, meta, sub = _train_fusion_for_combo(
            combo_name,
            combinations=combinations,
            combo_meta=combo_meta,
            long_df=long_df,
            subject_col=subject_col,
            device=device,
            train_kwargs=train_kwargs,
        )
        wrapper = _FusionProgWrapper(model)

        n_rows = arrays['struct'].shape[0]
        idx_bg = _sample_idx(n_rows, background_size, rng)
        idx_ev = _sample_idx(n_rows, eval_size, rng)

        bg_inputs = [arrays['struct'][idx_bg], arrays['diff'][idx_bg], arrays['back'][idx_bg]]
        ev_inputs = [arrays['struct'][idx_ev], arrays['diff'][idx_ev], arrays['back'][idx_ev]]

        explainer = _shap.DeepExplainer(wrapper, bg_inputs)
        shap_values = _normalize_deep_shap_values(_shap_values_safe(explainer, ev_inputs))

        struct_names = [feats[i] for i in meta['struct_idx']]
        diff_names = [feats[i] for i in meta['diff_idx']]
        back_names = [feats[i] for i in meta['back_idx']]

        struct_mean = _mean_shap_per_feature(shap_values[0]) if len(struct_names) else np.array([])
        diff_mean = _mean_shap_per_feature(shap_values[1]) if len(diff_names) else np.array([])
        back_mean = _mean_shap_per_feature(shap_values[2]) if len(back_names) else np.array([])

        modality_summary = {
            'struct': float(struct_mean.sum()) if struct_mean.size else 0.0,
            'diff': float(diff_mean.sum()) if diff_mean.size else 0.0,
            'back': float(back_mean.sum()) if back_mean.size else 0.0,
        }

        return {
            "model_kind": "fusion",
            "combo_name": combo_name,
            "feats": feats,
            "meta": meta,
            "modality_summary": modality_summary,
            "struct_names": struct_names,
            "diff_names": diff_names,
            "back_names": back_names,
            "struct_mean": struct_mean,
            "diff_mean": diff_mean,
            "back_mean": back_mean,
        }

    elif model_kind == "pair":
        from ..training.pair import _train_pair_for_combo, _PairProgWrapper

        model, (pair_x1, pair_x2), feats, meta, sub, sids = _train_pair_for_combo(
            combo_name,
            combinations=combinations,
            combo_meta=combo_meta,
            long_df=long_df,
            subject_col=subject_col,
            device=device,
            train_kwargs=train_kwargs,
        )
        wrapper = _PairProgWrapper(model)

        n_pairs = pair_x1.shape[0]
        idx_bg = _sample_idx(n_pairs, background_size, rng)
        idx_ev = _sample_idx(n_pairs, eval_size, rng)

        bg_inputs = [pair_x1[idx_bg], pair_x2[idx_bg]]
        ev_inputs = [pair_x1[idx_ev], pair_x2[idx_ev]]

        explainer = _shap.DeepExplainer(wrapper, bg_inputs)
        shap_values = _normalize_deep_shap_values(_shap_values_safe(explainer, ev_inputs))

        mean_x1 = _mean_shap_per_feature(shap_values[0])
        mean_x2 = _mean_shap_per_feature(shap_values[1])

        feat_mean = mean_x1 + mean_x2

        back_idx = np.array(meta['back_idx'], dtype=int)
        struct_idx = np.array(meta['struct_idx'], dtype=int)
        diff_idx = np.array(meta['diff_idx'], dtype=int)

        modality_summary = {
            'struct': float(feat_mean[struct_idx].sum()) if struct_idx.size else 0.0,
            'diff': float(feat_mean[diff_idx].sum()) if diff_idx.size else 0.0,
            'back': float(feat_mean[back_idx].sum()) if back_idx.size else 0.0,
        }

        return {
            "model_kind": "pair",
            "combo_name": combo_name,
            "feats": feats,
            "meta": meta,
            "modality_summary": modality_summary,
            "feat_mean": feat_mean,
            "back_names": [feats[i] for i in back_idx],
            "struct_names": [feats[i] for i in struct_idx],
            "diff_names": [feats[i] for i in diff_idx],
            "back_mean": feat_mean[back_idx] if back_idx.size else np.array([]),
            "struct_mean": feat_mean[struct_idx] if struct_idx.size else np.array([]),
            "diff_mean": feat_mean[diff_idx] if diff_idx.size else np.array([]),
        }

    raise ValueError(f"Unknown model_kind: {model_kind!r}")


__all__ = [
    "SHAP_BACKGROUND_SIZE",
    "SHAP_EVAL_SIZE",
    "TOP_N_FEATURES",
    "SHAP_CHECK_ADDITIVITY",
    "_as_numpy",
    "_squeeze_last_if_unit",
    "_mean_shap_per_feature",
    "_normalize_deep_shap_values",
    "_shap_values_safe",
    "_sample_idx",
    "_has_all_modalities",
    "_pick_best_combo",
    "run_shap_on_combo",
]
