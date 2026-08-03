"""Raw-file readers and per-file ses-3 helpers.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ids import find_id_column


# ---------------------------------------------------------------------------
# Spinal cord CSA/ECC filename helpers
# ---------------------------------------------------------------------------
def extract_subject_from_name(val):
    m = re.search(r"(sub-campac\d+)", str(val))
    return m.group(1) if m else None


def is_ses3(val):
    return "ses-3" in str(val)


# ---------------------------------------------------------------------------
# Raw file loader for Campinas follow-up mapping
# ---------------------------------------------------------------------------
def load_file_df(fname, raw_dir: Path):
    """Load a raw file by name. ROI CSV gets bilateral-average columns added.

    ROI CSV files receive bilateral-average peduncle columns. ``ROI_SCALE``
    remains 1.0 because the upstream ROI export is already used in its native
    unit for this merge path.
    """
    path = raw_dir / fname
    if fname.endswith('.csv'):
        df = pd.read_csv(path)
        # For ROI files, compute bilateral average peduncle volumes.
        if fname == "ROI_vbcb_p50_Vwm.csv":
            df.columns = df.columns.str.strip()
            ROI_SCALE = 1.0
            if "rSCP" in df.columns and "lSCP" in df.columns:
                df["scp_vol"] = ((df["rSCP"] + df["lSCP"]) / 2.0) * ROI_SCALE
            if "rMCP" in df.columns and "lMCP" in df.columns:
                df["mcp_vol"] = ((df["rMCP"] + df["lMCP"]) / 2.0) * ROI_SCALE
            if "rICP" in df.columns and "lICP" in df.columns:
                df["icp_vol"] = ((df["rICP"] + df["lICP"]) / 2.0) * ROI_SCALE
        return df
    return pd.read_excel(path)


# ---------------------------------------------------------------------------
# Melbourne demographic identifier helpers
# ---------------------------------------------------------------------------
def find_col(name_to_idx, *names):
    for name in names:
        name = name.lower()
        if name in name_to_idx:
            return name_to_idx[name]
    return None


def is_melfrd_clinical_id(val):
    try:
        v = float(val)
        return 54.0 <= v < 55.0
    except Exception:
        return False


def clinical_id_to_bids(cid):
    suffix = round(float(cid) * 100) - 5400
    return f"sub-melfrd{suffix:03d}"
