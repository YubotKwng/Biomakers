"""Subject ID parsing and inter-cohort reconciliation.
"""
from __future__ import annotations

import re


def std_col(name: str) -> str:
    # Standardize column names for matching
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_id_column(df):
    # Heuristic: prefer columns with 'bids' or 'subject' or 'id'
    candidates = [c for c in df.columns if re.search(r"bids|subject|id", str(c), re.I)]
    return candidates[0] if candidates else df.columns[0]


def parse_bids_subject_session(text):
    m = re.search(r"(sub-[^_]+)_(ses-\d+)", str(text))
    if m:
        return m.group(1), m.group(2)
    return None, None


def campac_to_pac(sub_id: str) -> str:
    # sub-campac01 -> pac01
    m = re.search(r"campac(\d+)", str(sub_id))
    if not m:
        return None
    return f"pac{int(m.group(1)):02d}"


def pac_to_campac(pac: str) -> str:
    # pac01 -> sub-campac01
    m = re.search(r"(\d+)", str(pac))
    if not m:
        return None
    return f"sub-campac{int(m.group(1)):02d}"
