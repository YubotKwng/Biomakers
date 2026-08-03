"""Training utilities for ``FusionModel`` with a paired-progression objective.

The routines prepare modality-specific tensors, optimize progression
sensitivity with optional clinical auxiliary heads, and expose trained models
for SHAP-based interpretation.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from ..data.qc import filter_complete_pairs
from ..eval.cv import split_train_val_subjects
from ..models.fusion import FusionModel
from .loss import paired_progression_loss


def prepare_fusion_arrays(df, feature_cols, meta, scaler, *, subject_col, device):
    """Build per-modality torch tensors + bookkeeping arrays.

    Returns tensors for modality branches, clinical targets, visit labels, and
    subject identifiers.
    """
    X = scaler.transform(df[feature_cols].values)
    arrays = {
        'struct': torch.tensor(X[:, meta['struct_idx']], dtype=torch.float32, device=device),
        'diff': torch.tensor(X[:, meta['diff_idx']], dtype=torch.float32, device=device),
        'back': torch.tensor(X[:, meta['back_idx']], dtype=torch.float32, device=device),
        'fars': torch.tensor(df['FARS'].values.reshape(-1, 1), dtype=torch.float32, device=device),
        'sara': torch.tensor(df['SARA'].values.reshape(-1, 1), dtype=torch.float32, device=device),
        'visit': np.array(df['visit'].values),
        'subject': np.array(df[subject_col].values),
    }
    return arrays


def evaluate_fusion_loss(
    model,
    arrays,
    *,
    use_clinical_heads=True,
    lambda_prog=1.0,
    lambda_fars=1.0,
    lambda_sara=1.0,
):
    """Compute multitask training / validation loss for FusionModel.

    Combines progression sensitivity with optional clinical reconstruction
    losses for FARS and SARA.
    """
    fars, sara, prog = model(arrays['struct'], arrays['diff'], arrays['back'])
    if use_clinical_heads:
        loss_fars = F.mse_loss(fars, arrays['fars'])
        loss_sara = F.mse_loss(sara, arrays['sara'])
    else:
        loss_fars = torch.tensor(0.0, device=fars.device)
        loss_sara = torch.tensor(0.0, device=fars.device)
    loss_prog = paired_progression_loss(prog.view(-1), arrays['visit'], arrays['subject'])
    total_loss = lambda_prog * loss_prog + lambda_fars * loss_fars + lambda_sara * loss_sara
    return total_loss, fars, sara, prog


def train_fusion_model(
    train_df,
    feature_cols,
    meta,
    scaler,
    *,
    subject_col,
    device,
    epochs=20,
    patience=4,
    lr=1e-3,
    weight_decay=1e-4,
    dropout=0.2,
    val_fraction=0.2,
    seed=42,
    use_clinical_heads=True,
    lambda_prog=1.0,
    lambda_fars=1.0,
    lambda_sara=1.0,
):
    """LOO-fold training loop with subject-level validation early stopping.

    Returns tensors for modality branches, clinical targets, visit labels, and
    subject identifiers.
    """
    train_split, val_split = split_train_val_subjects(
        train_df, subject_col, val_fraction=val_fraction, seed=seed
    )
    train_arrays = prepare_fusion_arrays(
        train_split, feature_cols, meta, scaler,
        subject_col=subject_col, device=device,
    )
    val_arrays = prepare_fusion_arrays(
        val_split, feature_cols, meta, scaler,
        subject_col=subject_col, device=device,
    )

    dims = {
        'struct': len(meta['struct_idx']),
        'diff': len(meta['diff_idx']),
        'back': len(meta['back_idx']),
    }
    model = FusionModel(dims, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    n_bad = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        train_loss, _, _, _ = evaluate_fusion_loss(
            model, train_arrays,
            use_clinical_heads=use_clinical_heads,
            lambda_prog=lambda_prog,
            lambda_fars=lambda_fars,
            lambda_sara=lambda_sara,
        )
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.inference_mode():
            val_loss, _, _, _ = evaluate_fusion_loss(
                model, val_arrays,
                use_clinical_heads=use_clinical_heads,
                lambda_prog=lambda_prog,
                lambda_fars=lambda_fars,
                lambda_sara=lambda_sara,
            )
        val_value = float(val_loss.item())

        if val_value < best_val:
            best_val = val_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            n_bad = 0
        else:
            n_bad += 1
        if n_bad >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch


# ---------------------------------------------------------------------------
# SHAP plumbing helpers.
# ---------------------------------------------------------------------------
class _FusionProgWrapper(nn.Module):
    """Forward only the progression output; used by ``shap.DeepExplainer``.

    Used by SHAP to explain only the progression output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_struct, x_diff, x_back):
        return self.model(x_struct, x_diff, x_back)[2]


def _train_fusion_for_combo(
    combo_name,
    *,
    combinations,
    combo_meta,
    long_df,
    subject_col,
    device,
    train_kwargs=None,
):
    """Train a FusionModel on ALL paired subjects of one combo, return
    everything SHAP needs.

    Trains a fusion model on all complete pairs for one modality combination
    and returns the artifacts required for SHAP.
    """
    combo = next(c for c in combinations if c['name'] == combo_name)
    feats = combo['features']
    meta = combo_meta[combo_name]
    sub = filter_complete_pairs(long_df, subject_col, feats)

    scaler = StandardScaler().fit(sub[feats].values)
    train_kwargs = dict(train_kwargs or {})
    model, _ = train_fusion_model(
        sub, feats, meta, scaler,
        subject_col=subject_col,
        device=device,
        **train_kwargs,
    )
    model.eval()

    arrays = prepare_fusion_arrays(
        sub, feats, meta, scaler,
        subject_col=subject_col, device=device,
    )
    return model, arrays, feats, meta, sub


__all__ = [
    "prepare_fusion_arrays",
    "evaluate_fusion_loss",
    "train_fusion_model",
    "_FusionProgWrapper",
    "_train_fusion_for_combo",
]
