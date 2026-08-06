"""Subject ID parsing and inter-cohort reconciliation.

These helpers keep ID handling in one place so merge code can compare
participants across BIDS-style imaging exports, CAMPAC names, and clinical
spreadsheet identifiers without repeating fragile regular expressions.
"""
from __future__ import annotations

import re


def std_col(name: str) -> str:
    """Normalise a spreadsheet column name for case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_id_column(df):
    """Return the most likely participant identifier column in ``df``.

    Raw files differ in naming convention, so this intentionally uses a small
    heuristic instead of requiring one exact column name.
    """
    candidates = [c for c in df.columns if re.search(r"bids|subject|id", str(c), re.I)]
    return candidates[0] if candidates else df.columns[0]


def parse_bids_subject_session(text):
    """Extract ``(sub-..., ses-...)`` from a BIDS-like filename or path."""
    m = re.search(r"(sub-[^_]+)_(ses-\d+)", str(text))
    if m:
        return m.group(1), m.group(2)
    return None, None


def campac_to_pac(sub_id: str) -> str:
    """Convert BIDS-style CAMPAC subject IDs to clinical ``pac`` IDs."""
    m = re.search(r"campac(\d+)", str(sub_id))
    if not m:
        return None
    return f"pac{int(m.group(1)):02d}"


def pac_to_campac(pac: str) -> str:
    """Convert clinical ``pac`` IDs back to BIDS-style CAMPAC subject IDs."""
    m = re.search(r"(\d+)", str(pac))
    if not m:
        return None
    return f"sub-campac{int(m.group(1)):02d}"
