"""Training utilities for ``PairModel`` using a shared-encoder SRM objective."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from ..data.qc import filter_complete_pairs
from ..data.model_safety import assert_no_control_rows, assert_no_clinical_score_features
from ..eval.cv import split_train_val_subjects
from ..models.pair import PairModel


def prepare_pair_arrays(df, feature_cols, scaler, *, subject_col, device, include_clinical_targets=False):
    """Build paired-tensor inputs (visit-1 and visit-2 features per subject).

    Clinical target tensors are zero-filled unless explicitly requested.
    """
    assert_no_control_rows(df)
    assert_no_clinical_score_features(feature_cols)
    x1_list, x2_list, sids = [], [], []
    for sid, g in df.groupby(subject_col):
        # PairModel needs exactly one baseline and one follow-up row per
        # interval/subject key so progression is well-defined.
        g = g.sort_values('visit')
        if g['visit'].nunique() != 2:
            continue
        x1 = g[g['visit'] == 1][feature_cols].values
        x2 = g[g['visit'] == 2][feature_cols].values
        if len(x1) == 1 and len(x2) == 1:
            x1_list.append(x1.ravel())
            x2_list.append(x2.ravel())
            sids.append(sid)
    # The scaler is fit outside this helper on the current training fold.
    X1 = torch.tensor(scaler.transform(np.vstack(x1_list)), dtype=torch.float32, device=device)
    X2 = torch.tensor(scaler.transform(np.vstack(x2_list)), dtype=torch.float32, device=device)
    if include_clinical_targets:
        F1 = np.array([df[df[subject_col] == sid].sort_values("visit")["FARS"].iloc[0] for sid in sids], dtype=float)
        F2 = np.array([df[df[subject_col] == sid].sort_values("visit")["FARS"].iloc[1] for sid in sids], dtype=float)
        S1 = np.array([df[df[subject_col] == sid].sort_values("visit")["SARA"].iloc[0] for sid in sids], dtype=float)
        S2 = np.array([df[df[subject_col] == sid].sort_values("visit")["SARA"].iloc[1] for sid in sids], dtype=float)
    else:
        F1 = np.zeros((len(sids),), dtype=float)
        F2 = np.zeros((len(sids),), dtype=float)
        S1 = np.zeros((len(sids),), dtype=float)
        S2 = np.zeros((len(sids),), dtype=float)
    F1 = torch.tensor(F1.reshape(-1, 1), dtype=torch.float32, device=device)
    F2 = torch.tensor(F2.reshape(-1, 1), dtype=torch.float32, device=device)
    S1 = torch.tensor(S1.reshape(-1, 1), dtype=torch.float32, device=device)
    S2 = torch.tensor(S2.reshape(-1, 1), dtype=torch.float32, device=device)
    return X1, X2, np.array(sids), F1, F2, S1, S2


def train_pair_model(
    train_df,
    feature_cols,
    scaler,
    *,
    subject_col,
    split_group_col=None,
    device,
    epochs=20,
    patience=4,
    lr=1e-3,
    weight_decay=1e-4,
    dropout=0.2,
    val_fraction=0.2,
    seed=42,
    use_clinical_heads=False,
    lambda_prog=1.0,
    lambda_fars=0.0,
    lambda_sara=0.0,
):
    """Train ``PairModel`` with subject-level early stopping.

    Uses subject-level validation for early stopping and optimizes paired
    progression sensitivity. Clinical auxiliary heads are disabled by default
    because clinical scores must not be used during model training.
    """
    if use_clinical_heads:
        raise ValueError("Clinical auxiliary heads use FARS/SARA during training and are disabled by policy.")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    assert_no_control_rows(train_df)
    assert_no_clinical_score_features(feature_cols)
    # Validation is selected from the current training fold only; it is used
    # for early stopping, not for final OOF evaluation.
    train_split, val_split = split_train_val_subjects(
        train_df,
        subject_col,
        val_fraction=val_fraction,
        seed=seed,
        split_group_col=split_group_col,
    )
    X1_train, X2_train, sid_train, F1_train, F2_train, S1_train, S2_train = prepare_pair_arrays(
        train_split, feature_cols, scaler, subject_col=subject_col, device=device,
        include_clinical_targets=use_clinical_heads,
    )
    X1_val, X2_val, sid_val, F1_val, F2_val, S1_val, S2_val = prepare_pair_arrays(
        val_split, feature_cols, scaler, subject_col=subject_col, device=device,
        include_clinical_targets=use_clinical_heads,
    )

    model = PairModel(X1_train.shape[1], dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    n_bad = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        prog, _, _, _, _ = model(X1_train, X2_train)
        # Optimise negative SRM: maximising mean(progression)/sd(progression)
        # across training pairs while keeping the model unsupervised by clinic.
        train_loss = -(prog.mean() / (prog.std(unbiased=True) + 1e-6))
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.inference_mode():
            prog_val, fars1_v, fars2_v, sara1_v, sara2_v = model(X1_val, X2_val)
            prog_loss_v = -(prog_val.mean() / (prog_val.std(unbiased=True) + 1e-6))
            if use_clinical_heads:
                loss_fars_v = F.mse_loss(fars1_v, F1_val) + F.mse_loss(fars2_v, F2_val)
                loss_sara_v = F.mse_loss(sara1_v, S1_val) + F.mse_loss(sara2_v, S2_val)
            else:
                loss_fars_v = torch.tensor(0.0, device=device)
                loss_sara_v = torch.tensor(0.0, device=device)
            val_loss = lambda_prog * prog_loss_v + lambda_fars * loss_fars_v + lambda_sara * loss_sara_v
        val_value = float(val_loss.item())

        # Select the checkpoint with the strongest validation progression
        # objective, again without clinical-score losses by default.
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
class _PairProgWrapper(nn.Module):
    """Forward only the progression scalar output; used by SHAP.

    Used by SHAP to explain only the progression output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x1, x2):
        """Forward paired visits and keep only the progression output."""
        return self.model(x1, x2)[0]


def _build_pair_tensors(sub_df, feats, scaler, *, subject_col, device):
    """Construct (x1, x2, sids) torch tensors from a long-format subset.

    Used by SHAP to explain only the progression output.
    """
    x1_list, x2_list, sid_list = [], [], []
    for sid, g in sub_df.groupby(subject_col):
        g = g.sort_values('visit')
        if g['visit'].nunique() != 2:
            continue
        x1 = scaler.transform(g[g['visit'] == 1][feats].values)
        x2 = scaler.transform(g[g['visit'] == 2][feats].values)
        if x1.shape[0] != 1 or x2.shape[0] != 1:
            continue
        x1_list.append(x1[0])
        x2_list.append(x2[0])
        sid_list.append(sid)

    X1 = np.asarray(x1_list, dtype=np.float32)
    X2 = np.asarray(x2_list, dtype=np.float32)
    return (
        torch.tensor(X1, dtype=torch.float32, device=device),
        torch.tensor(X2, dtype=torch.float32, device=device),
        np.asarray(sid_list),
    )


def _train_pair_for_combo(
    combo_name,
    *,
    combinations,
    combo_meta,
    long_df,
    subject_col,
    device,
    train_kwargs=None,
):
    """Train PairModel on all paired subjects of one combo, return
    everything SHAP needs.

    Trains a pair model on all complete pairs for one modality combination
    and returns the artifacts required for SHAP.
    """
    combo = next(c for c in combinations if c['name'] == combo_name)
    feats = combo['features']
    sub = filter_complete_pairs(long_df, subject_col, feats)

    scaler = StandardScaler().fit(sub[feats].values)
    train_kwargs = dict(train_kwargs or {})
    model, _ = train_pair_model(
        sub, feats, scaler,
        subject_col=subject_col, device=device, **train_kwargs,
    )
    model.eval()

    x1, x2, sids = _build_pair_tensors(
        sub, feats, scaler, subject_col=subject_col, device=device
    )
    return model, (x1, x2), feats, combo_meta[combo_name], sub, sids


__all__ = [
    "prepare_pair_arrays",
    "train_pair_model",
    "_PairProgWrapper",
    "_build_pair_tensors",
    "_train_pair_for_combo",
]
