"""Helpers for working with TRACK-FA paired (V1/V2) CSV exports.

This repo's merge pipeline produces `trackfa_pairs*.csv` in a *paired wide*
format: one row per subject, columns suffixed with `_baseline` / `_followup`,
and optionally `delta_*` columns.

The baseline-paper reproduction code expects a *long* format with one row per
(subject, visit), a `visit` column ∈ {1, 2}, and per-visit targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .reshape import wide_to_long


_PATIENT_ID_RE = re.compile(r"^(?P<subj>[A-Za-z0-9]+)_(?P<pair>V\\d+V\\d+)$")


def parse_trackfa_patient_id(patient_id: str) -> str:
    """Return subject id from `patient_id` like `AAN001_V1V2` → `AAN001`."""
    s = str(patient_id)
    m = _PATIENT_ID_RE.match(s)
    if not m:
        # Fall back to a conservative split rather than raising: downstream
        # GroupKFold/LOO only needs stable subject ids.
        return s.split("_")[0]
    return m.group("subj")


def coerce_gaa_repeat(value) -> float:
    """Parse `gaa_*` values that may include non-numeric tokens.

    Examples observed in TRACK-FA exports:
    - '0775' → 775
    - ' >700' → 700
    - '' / NaN → NaN
    """
    if value is None:
        return float("nan")
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    # Handle strings like '>700' / ' >700'
    s = s.replace(">", "").strip()
    try:
        return float(s)
    except ValueError:
        # Last resort: pull the first number substring.
        m = re.search(r"(\d+(\.\d+)?)", s)
        return float(m.group(1)) if m else float("nan")


def trackfa_pairs_to_long(
    pairs_df: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
    baseline_suffix: str = "_baseline",
    followup_suffix: str = "_followup",
) -> pd.DataFrame:
    """Convert a paired wide TRACK-FA table into long format.

    Output columns:
    - `subject` (parsed from `patient_id`)
    - `visit` ∈ {1, 2}
    - background/static columns copied to both visits (if present)
    - per-visit targets `FARS`, `SARA`, `ADL` (if present in the wide table)
    - per-visit imaging features (base names without suffixes)
    """
    if patient_id_col not in pairs_df.columns:
        raise KeyError(f"Missing required column: {patient_id_col!r}")

    wide = pairs_df.copy()
    wide["subject"] = wide[patient_id_col].apply(parse_trackfa_patient_id)
    # Pair identifier: keep the full patient_id so V1V2 and V2V3 remain distinct.
    wide["pair_id"] = wide[patient_id_col].astype(str)

    # Coerce gaa repeats to numeric if present.
    for c in ("gaa_1", "gaa_2"):
        if c in wide.columns:
            wide[c] = wide[c].map(coerce_gaa_repeat)

    # Build long-format per-visit columns from *_baseline / *_followup pairs.
    extra_targets: List[Tuple[str, str, str]] = []
    if "mfars_total_baseline" in wide.columns and "mfars_total_followup" in wide.columns:
        extra_targets.append(("FARS", "mfars_total_baseline", "mfars_total_followup"))
    if "sara_total_baseline" in wide.columns and "sara_total_followup" in wide.columns:
        extra_targets.append(("SARA", "sara_total_baseline", "sara_total_followup"))
    if "adl_total_baseline" in wide.columns and "adl_total_followup" in wide.columns:
        extra_targets.append(("ADL", "adl_total_baseline", "adl_total_followup"))

    long = wide_to_long(
        wide,
        subject_col="pair_id",
        suffixes=(baseline_suffix, followup_suffix),
        extra_targets=extra_targets,
    )

    # Attach subject id for CV grouping.
    long = long.merge(wide[["pair_id", "subject"]].drop_duplicates("pair_id"), on="pair_id", how="left")

    # Attach background/static covariates to every long row.
    static_cols = [
        c
        for c in (
            patient_id_col,
            "site",
            "age",
            "gender",
            "gaa_1",
            "gaa_2",
            "onset_age",
            "disease_duration",
        )
        if c in wide.columns
    ]
    if static_cols:
        long = long.merge(wide[["pair_id"] + static_cols].drop_duplicates("pair_id"), on="pair_id", how="left")

    # Prefer `sex` naming used by the baseline paper code paths.
    if "gender" in long.columns and "sex" not in long.columns:
        long = long.rename(columns={"gender": "sex"})

    return long


@dataclass(frozen=True)
class TrackfaFeatureGroups:
    """Feature groups derived from `trackfa_pairs_drop3poms.csv` columns."""

    background: List[str]
    clinical_deltas: List[str]
    poms: List[str]
    brainspinemorph: List[str]
    braindti: List[str]

    @property
    def all_neuroimaging(self) -> List[str]:
        return sorted(set(self.poms + self.brainspinemorph + self.braindti))


def infer_trackfa_feature_groups(pairs_df: pd.DataFrame) -> TrackfaFeatureGroups:
    """Infer modality lists (base names) from a paired wide TRACK-FA table."""
    cols = list(pairs_df.columns)

    # Base names appearing as *_baseline/_followup or delta_*.
    base: set[str] = set()
    for c in cols:
        if c.endswith("_baseline"):
            base.add(c[: -len("_baseline")])
        elif c.endswith("_followup"):
            base.add(c[: -len("_followup")])
        elif c.startswith("delta_"):
            base.add(c[len("delta_") :])

    has_gender = "gender" in cols
    background = []
    for c in ("age", "sex", "gaa_1", "gaa_2", "onset_age", "disease_duration"):
        if c in cols or c in base:
            background.append(c)
        elif c == "sex" and has_gender:
            # Normalised name used by the reproduction pipeline.
            background.append("sex")
    clinical_deltas = [c for c in ("delta_mfars_total", "delta_sara_total", "delta_adl_total") if c in cols]

    poms = [
        "csa_c1c2",
        "Cereb_vol",
        "SCP_vol",
        "sFA_c3c5",
        "sMD_c3c5",
        "sRD_c3c5",
        "sAD_c3c5",
        "FA_SCP",
        "MD_SCP",
        "RD_SCP",
        "AD_SCP",
        "TotalBrainGMVol_nocereb",
        "TotalBrainWMVol_nocereb",
        "TotalBrainVol_nocereb",
        "eTIV",
    ]
    poms_present = [b for b in poms if b in base]

    # DTI: 4 metrics × tracts.
    dti = sorted([b for b in base if re.match(r"^(FA|MD|RD|AD)_.+$", b)])

    clinical_scales = {"mfars_total", "sara_total", "adl_total"}
    remainder = sorted([b for b in base if b not in set(poms) and b not in set(dti) and b not in clinical_scales])

    return TrackfaFeatureGroups(
        background=background,
        clinical_deltas=clinical_deltas,
        poms=poms_present,
        brainspinemorph=remainder,
        braindti=dti,
    )


def build_trackfa_combinations(groups: TrackfaFeatureGroups) -> List[Dict]:
    """Return paper-style feature combinations for TRACK-FA."""
    bg = list(groups.background)
    poms = list(groups.poms)
    morph = list(groups.brainspinemorph)
    dti = list(groups.braindti)
    all_img = list(groups.all_neuroimaging)

    combos = [
        {"name": "background_only", "domains": [bg], "skip": False},
        {"name": "poms_only", "domains": [poms], "skip": False},
        {"name": "brainspinemorph_only", "domains": [morph], "skip": False},
        {"name": "braindti_only", "domains": [dti], "skip": False},
        {"name": "all_neuroimaging", "domains": [all_img], "skip": False},
        {"name": "background_poms", "domains": [bg, poms], "skip": False},
        {"name": "background_brainspinemorph", "domains": [bg, morph], "skip": False},
        {"name": "background_braindti", "domains": [bg, dti], "skip": False},
        {"name": "background_all_neuroimaging", "domains": [bg, all_img], "skip": False},
    ]
    return combos


__all__ = [
    "parse_trackfa_patient_id",
    "coerce_gaa_repeat",
    "trackfa_pairs_to_long",
    "TrackfaFeatureGroups",
    "infer_trackfa_feature_groups",
    "build_trackfa_combinations",
]
