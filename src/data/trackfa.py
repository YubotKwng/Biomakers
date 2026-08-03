"""TRACK-FA preprocessing (REDCap export + imaging MasterFile) → wide patient CSV.

The output shape is 1 row per participant (``ID``) with visit-suffixed columns
(``*_v1``, ``*_v2``, ``*_v3``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config, DEFAULT_CONFIG
from .ids import std_col


EVENT_ORDER = {
    "enrolment_arm_1": 0,
    "study_visit_1_arm_1": 1,
    "study_visit_2_arm_1": 2,
    "study_visit_3_arm_1": 3,
}

DEFAULT_CLINICAL_DROP_COLUMNS = {
    # Education/school fields
    "education_years",
    "at_school",
    "school_grade",
    "education_highest_level",
    "education_isced",
    # GAA repeat confirmation + lab detail free-text (not analytic features)
    "prev_lab_details",
    "new_lab_details",
}

DEFAULT_CLINICAL_DROP_PREFIXES = (
    # REDCap instrument completion/status fields
    "sop_",
    # Checkbox expansions for repeat_source (source of GAA confirmation)
    "repeat_source___",
)

DEFAULT_CLINICAL_DROP_SUFFIXES = (
    # REDCap instrument completion/status fields end with _complete
    "_complete",
)

DEFAULT_TRACKFA_CLINICAL_KEEP_COLUMNS = (
    # IDs / grouping
    "ID",
    "site",
    "study_group",
    # Demographics / baseline
    "age",
    "gender",
    "handedness_score",
    # Genetics / onset / history
    "gaa_1",
    "gaa_2",
    "point_mut",
    "point_mut_det",
    "onset_age",
    "first_symptom",
    "disease_duration",
    # Biospecimens
    "upenn_fxn_total",
)


def _expand_visit_cols(pattern: str, visits: Iterable[int]) -> list[str]:
    return [pattern.format(v=v) for v in visits]


def select_trackfa_clinical_features(
    clinical_df: pd.DataFrame,
    *,
    visits: Tuple[int, int, int] = (1, 2, 3),
    vygr_visits: Tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Select the analytic TRACK-FA REDCap clinical feature set (45 columns).

    Keeps exactly the requested columns (when present), dropping everything else.
    """
    keep = list(DEFAULT_TRACKFA_CLINICAL_KEEP_COLUMNS)
    keep += _expand_visit_cols("bmi_v{v}", visits)
    keep += _expand_visit_cols("functional_staging_score_v{v}", visits)
    keep += _expand_visit_cols("mfars_total_v{v}", visits)
    keep += _expand_visit_cols("adl_total_v{v}", visits)
    keep += _expand_visit_cols("sara_total_v{v}", visits)
    keep += _expand_visit_cols("hpt_dom_av_v{v}", visits)
    keep += _expand_visit_cols("hpt_ndom_av_v{v}", visits)
    keep += _expand_visit_cols("lcslc_total_v{v}", visits)
    keep += _expand_visit_cols("scan_date_crf_v{v}", visits)
    keep += [f"vygr_nfl_conc_{k}_v{v}" for k in (1, 2) for v in vygr_visits]

    keep_set = set(keep)
    cols = [c for c in clinical_df.columns if c in keep_set]

    # Ensure ID is present if expected.
    if "ID" not in cols and "ID" in keep_set:
        raise ValueError("Expected collapsed clinical_df to contain an 'ID' column.")

    return clinical_df.loc[:, cols].copy()

AUDIT_COLUMNS = [
    "timestamp_utc",
    "step",
    "reason",
    "n_before",
    "n_after",
    "n_removed",
    "participant_id",
    "column",
    "details_json",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return json.dumps({"unserializable": str(obj)}, ensure_ascii=False)


def _add_audit(
    rows: list[dict[str, Any]],
    *,
    step: str,
    reason: str,
    n_before: int,
    n_after: int,
    participant_id: Optional[str] = None,
    column: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    rows.append(
        {
            "timestamp_utc": _utc_now_iso(),
            "step": step,
            "reason": reason,
            "n_before": int(n_before),
            "n_after": int(n_after),
            "n_removed": int(n_before - n_after),
            "participant_id": participant_id,
            "column": column,
            "details_json": _safe_json_dumps(details or {}),
        }
    )


def load_trackfa_redcap_export(path: Path) -> pd.DataFrame:
    """Load the REDCap export CSV (TRACK-FA)."""
    df = pd.read_csv(path)
    # Keep source names; provide standardized view for matching/debugging.
    df.attrs["std_columns"] = {c: std_col(c) for c in df.columns}
    return df


def _event_rank(name: Any) -> int:
    s = str(name)
    if s in EVENT_ORDER:
        return EVENT_ORDER[s]
    # fallback: try to infer visit number
    if "enrol" in s.lower():
        return 0
    for k in (1, 2, 3):
        if f"visit_{k}" in s.lower() or f"visit{k}" in s.lower():
            return k
    return 999


def _resolve_mode_with_tiebreak(
    values: pd.Series,
    event_names: pd.Series,
) -> Any:
    """Resolve a set of values by mode; tie-break by earliest event order."""
    # values and event_names are aligned within a group.
    non_missing = [(v, e) for v, e in zip(values.tolist(), event_names.tolist()) if not _is_missing(v)]
    if not non_missing:
        return np.nan

    vals = pd.Series([v for v, _ in non_missing])
    counts = vals.value_counts(dropna=False)
    top = counts.max()
    winners = set(counts[counts == top].index.tolist())
    if len(winners) == 1:
        return next(iter(winners))

    # tie-break: first appearance by earliest event rank
    best_val = None
    best_rank = 10**9
    for v, e in non_missing:
        if v not in winners:
            continue
        r = _event_rank(e)
        if r < best_rank:
            best_rank = r
            best_val = v
    return best_val


def collapse_trackfa_events(
    redcap_df: pd.DataFrame, *, id_col: str = "participant_id"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse 4 event rows per participant → 1 row per participant.

    Rules:
    - For each column: take first non-null across events.
    - If a column has >1 distinct non-null values: resolve by mode, tie-break by
      earliest event order and emit an audit row.
    - Output uses ``ID`` as the identifier column.
    """
    audit_rows: list[dict[str, Any]] = []

    df = redcap_df.copy()
    n0 = len(df)
    df[id_col] = df.get(id_col)
    missing_id = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
    if missing_id.any():
        df = df[~missing_id].copy()
        _add_audit(
            audit_rows,
            step="collapse_events",
            reason="drop_missing_participant_id",
            n_before=n0,
            n_after=len(df),
            details={"id_col": id_col},
        )

    # Stable event order for tie-breaking.
    event_col = "redcap_event_name" if "redcap_event_name" in df.columns else None
    if event_col is not None:
        df["_event_rank"] = df[event_col].apply(_event_rank)
    else:
        df["_event_rank"] = 999

    # Ignore the REDCap bookkeeping columns for aggregation.
    ignore = {
        id_col,
        "redcap_event_name",
        "redcap_repeat_instrument",
        "redcap_repeat_instance",
        "_event_rank",
    }

    out_rows: list[dict[str, Any]] = []
    for pid, g in df.groupby(id_col, sort=True):
        g = g.sort_values("_event_rank", kind="stable")
        row: dict[str, Any] = {"ID": pid}
        event_names = g[event_col] if event_col is not None else pd.Series([None] * len(g), index=g.index)

        for col in df.columns:
            if col in ignore:
                continue
            series = g[col]
            non_missing = series[~series.isna()]
            if non_missing.empty:
                row[col] = np.nan
                continue

            # Distinct non-null values
            distinct = pd.unique(non_missing)
            distinct = [v for v in distinct if not _is_missing(v)]
            if len(distinct) <= 1:
                # first non-null in event order
                first_idx = next((i for i, v in zip(series.index.tolist(), series.tolist()) if not _is_missing(v)), None)
                row[col] = series.loc[first_idx] if first_idx is not None else np.nan
                continue

            resolved = _resolve_mode_with_tiebreak(series, event_names)
            row[col] = resolved
            _add_audit(
                audit_rows,
                step="collapse_events",
                reason="conflict_resolved",
                n_before=len(g),
                n_after=1,
                participant_id=str(pid),
                column=col,
                details={
                    "distinct_values": [str(v) for v in distinct[:20]],
                    "n_distinct": len(distinct),
                },
            )

        out_rows.append(row)

    wide = pd.DataFrame(out_rows)
    # Drop helper if it leaked.
    if "_event_rank" in wide.columns:
        wide = wide.drop(columns=["_event_rank"])

    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return wide, audit


def drop_trackfa_clinical_metadata_columns(
    clinical_df: pd.DataFrame,
    *,
    drop_columns: Iterable[str] = DEFAULT_CLINICAL_DROP_COLUMNS,
    drop_prefixes: Tuple[str, ...] = DEFAULT_CLINICAL_DROP_PREFIXES,
    drop_suffixes: Tuple[str, ...] = DEFAULT_CLINICAL_DROP_SUFFIXES,
) -> pd.DataFrame:
    """Drop REDCap metadata / QC fields from the collapsed clinical table.

    This targets:
    - REDCap instrument status fields like ``sop_*_complete``.
    - Checkbox expansions like ``repeat_source___1``.
    - Education/school fields requested to remove from modeling.
    """
    df = clinical_df.copy()
    drop_set = {str(c) for c in drop_columns}

    cols_to_drop: list[str] = []
    for c in df.columns:
        sc = str(c)
        if sc in drop_set:
            cols_to_drop.append(c)
            continue
        if sc.startswith("repeat_source___"):
            cols_to_drop.append(c)
            continue
        if any(sc.startswith(p) for p in drop_prefixes) and any(sc.endswith(s) for s in drop_suffixes):
            cols_to_drop.append(c)
            continue

    # De-dup while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols_to_drop:
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)

    return df.drop(columns=ordered, errors="ignore")


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Reading the TRACK-FA MasterFile (.xlsx) requires `openpyxl`. "
            "Install it (e.g. `pip install openpyxl`) and rerun. "
            f"Original error: {e}"
        ) from e


def load_trackfa_masterfile(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Load TRACK-FA imaging MasterFile Excel sheet.

    If ``sheet`` is None, auto-select the first sheet that appears to contain a
    TRACK-FA participant id column with values starting with ``TRACKFA_``.
    """
    _require_openpyxl()

    if sheet is not None:
        df = pd.read_excel(path, sheet_name=sheet)
        df.attrs["selected_sheet"] = sheet
        return df

    xl = pd.ExcelFile(path)
    for s in xl.sheet_names:
        try:
            preview = xl.parse(s, nrows=50)
        except Exception:
            continue
        pid_col = _infer_participant_col(preview)
        if pid_col is None:
            continue
        vals = preview[pid_col].astype(str).fillna("")
        if (vals.str.startswith("TRACKFA_") | vals.str.startswith("TRACK-FA_")).any():
            df = xl.parse(s)
            df.attrs["selected_sheet"] = s
            return df

    # Fall back to first sheet.
    df = xl.parse(xl.sheet_names[0])
    df.attrs["selected_sheet"] = xl.sheet_names[0]
    return df


def load_trackfa_masterfile_all_sheets(path: Path) -> pd.DataFrame:
    """Load and outer-merge all sheets from the TRACK-FA imaging MasterFile.

    Expected sheets include: POMs, BrainSpineMorph, BrainDTI.

    Merge key: [SubjectID, Visit, ScanDate, Protocol] when present.
    Keeps every feature column across sheets, except for known duplicated
    columns where we retain the POMs copy and drop the duplicates coming from
    BrainSpineMorph.
    """
    _require_openpyxl()

    xl = pd.ExcelFile(path)
    sheets: dict[str, pd.DataFrame] = {name: xl.parse(name) for name in xl.sheet_names}

    def _key_cols(df: pd.DataFrame) -> list[str]:
        candidates = ["SubjectID", "Visit", "ScanDate", "Protocol"]
        return [c for c in candidates if c in df.columns]

    merged: Optional[pd.DataFrame] = None
    for name in xl.sheet_names:
        df = sheets[name]
        if merged is None:
            merged = df.copy()
            continue
        keys = [c for c in _key_cols(merged) if c in _key_cols(df)]
        if not keys:
            raise ValueError(f"Cannot merge MasterFile sheet '{name}': no shared key columns found.")
        merged = merged.merge(df, on=keys, how="outer", suffixes=("", f"_{name.lower()}"))

    assert merged is not None

    # Drop known duplicates from BrainSpineMorph: keep POMs copy.
    morph_suffix = "_brainspinemorph"
    duplicated_keep_from_poms = [
        "TotalBrainGMVol_nocereb",
        "TotalBrainWMVol_nocereb",
        "TotalBrainVol_nocereb",
        "eTIV",
    ]
    drop_cols = [f"{c}{morph_suffix}" for c in duplicated_keep_from_poms if f"{c}{morph_suffix}" in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    merged.attrs["selected_sheet"] = "ALL"
    merged.attrs["sheets_loaded"] = list(xl.sheet_names)
    return merged


def _infer_participant_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer common names.
    for c in df.columns:
        sc = std_col(c)
        if sc in {
            "participant_id",
            "participant",
            "subject_id",
            "subjectid",
            "subject",
            "record_id",
            "recordid",
            "patient_id",
            "patientid",
            "id",
        }:
            return c
    # Heuristic: a column containing TRACKFA_* like values.
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.startswith("TRACKFA_").any():
            return c
    return None


def _infer_visit_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        sc = std_col(c)
        if sc in {"visit", "visit_number", "study_visit", "visitnum", "visit_no"}:
            return c
    return None


_TRACKFA_PREFIX_RE = re.compile(r"^(TRACKFA_|TRACK-FA_)", flags=re.IGNORECASE)
_TRACKFA_SHORT_ID_RE = re.compile(r"^[A-Z]{2,6}\d{2,6}$")


def _normalize_trackfa_participant_id(value: Any) -> Any:
    if _is_missing(value):
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan
    if _TRACKFA_PREFIX_RE.match(s):
        return s.replace("TRACK-FA_", "TRACKFA_")
    # Imaging masterfiles sometimes store SubjectID as a short site code (e.g., MEL024)
    # while REDCap uses TRACKFA_MEL024. Promote short ids into TRACKFA_*.
    if _TRACKFA_SHORT_ID_RE.match(s):
        return f"TRACKFA_{s}"
    return s


def normalize_trackfa_masterfile(master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize imaging master file into long format: (participant_id, visit, features...)."""
    audit_rows: list[dict[str, Any]] = []
    df = master_df.copy()
    n0 = len(df)

    pid_col = _infer_participant_col(df)
    if pid_col is None:
        raise ValueError("Could not infer participant id column in TRACK-FA MasterFile.")

    # Standardize participant id column name for downstream joins.
    df = df.rename(columns={pid_col: "participant_id"})
    df["participant_id"] = df["participant_id"].apply(_normalize_trackfa_participant_id)

    # Drop rows without participant_id.
    missing_pid = df["participant_id"].isna() | (df["participant_id"].astype(str).str.strip() == "")
    if missing_pid.any():
        df = df[~missing_pid].copy()
        _add_audit(
            audit_rows,
            step="normalize_masterfile",
            reason="drop_missing_participant_id",
            n_before=n0,
            n_after=len(df),
            details={"participant_col": pid_col},
        )

    # Wide vs long detection.
    cols = list(df.columns)
    has_suffix = any(str(c).endswith(("_v1", "_v2", "_v3")) for c in cols)
    visit_col = _infer_visit_col(df)

    if visit_col is not None and not has_suffix:
        # Long-ish already: map visit to int 1/2/3.
        def _to_visit(x: Any) -> float:
            if _is_missing(x):
                return np.nan
            s = str(x).strip().lower()
            for k in (1, 2, 3):
                if s == str(k) or f"visit {k}" in s or f"visit{k}" in s or f"v{k}" == s:
                    return float(k)
            try:
                v = float(x)
                if v in (1, 2, 3):
                    return float(v)
            except Exception:
                return np.nan
            return np.nan

        df["visit"] = df[visit_col].apply(_to_visit).astype("Int64")
        before = len(df)
        df = df[df["visit"].isin([1, 2, 3])].copy()
        _add_audit(
            audit_rows,
            step="normalize_masterfile",
            reason="drop_invalid_visit",
            n_before=before,
            n_after=len(df),
            details={"visit_col": visit_col},
        )

        # Keep participant_id, visit, and the rest of columns (except source visit column).
        keep = ["participant_id", "visit"] + [c for c in df.columns if c not in {"participant_id", "visit", visit_col}]
        df = df[keep]
        return df.reset_index(drop=True), pd.DataFrame(audit_rows)

    if not has_suffix:
        raise ValueError(
            "TRACK-FA MasterFile appears to have neither a visit column nor *_v1/_v2/_v3 columns."
        )

    # Wide: unpivot by suffix.
    suffixes = ("_v1", "_v2", "_v3")
    base_to_cols: dict[int, list[tuple[str, str]]] = {1: [], 2: [], 3: []}  # visit -> [(base, colname)]
    static_cols = [c for c in df.columns if c not in {"participant_id"} and not str(c).endswith(suffixes)]
    for c in df.columns:
        s = str(c)
        for k, sfx in enumerate(suffixes, start=1):
            if s.endswith(sfx):
                base = s[: -len(sfx)]
                base_to_cols[k].append((base, c))
                break

    long_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        pid = r["participant_id"]
        static_payload = {c: r.get(c) for c in static_cols}
        for visit in (1, 2, 3):
            row = {"participant_id": pid, "visit": visit, **static_payload}
            for base, col in base_to_cols[visit]:
                row[base] = r.get(col)
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    return long_df.reset_index(drop=True), pd.DataFrame(audit_rows)


def merge_trackfa(
    redcap_wide: pd.DataFrame, imaging_long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge patient-level REDCap wide with imaging long (participant_id, visit)."""
    audit_rows: list[dict[str, Any]] = []

    if "ID" not in redcap_wide.columns:
        raise ValueError("Expected `redcap_wide` to contain an `ID` column.")

    # Resolve imaging duplicates per (participant_id, visit)
    if not {"participant_id", "visit"}.issubset(set(imaging_long.columns)):
        raise ValueError("Expected imaging_long to have columns: participant_id, visit")

    img = imaging_long.copy()
    before = len(img)
    img["visit"] = pd.to_numeric(img["visit"], errors="coerce").astype("Int64")
    img = img[img["visit"].isin([1, 2, 3])].copy()
    _add_audit(
        audit_rows,
        step="merge_imaging",
        reason="drop_invalid_visit",
        n_before=before,
        n_after=len(img),
    )

    key = ["participant_id", "visit"]
    dup_mask = img.duplicated(subset=key, keep=False)
    if dup_mask.any():
        # keep row with max non-null feature fields
        feature_cols = [c for c in img.columns if c not in key]

        def _score(row: pd.Series) -> int:
            return int(row[feature_cols].notna().sum())

        img["_nn_score"] = img.apply(_score, axis=1)
        img = (
            img.sort_values(["participant_id", "visit", "_nn_score"], ascending=[True, True, False])
            .drop_duplicates(subset=key, keep="first")
            .drop(columns=["_nn_score"])
        )
        _add_audit(
            audit_rows,
            step="merge_imaging",
            reason="imaging_duplicate_resolved",
            n_before=int(dup_mask.sum()),
            n_after=int(img.duplicated(subset=key).sum()),
            details={"resolution": "keep_max_non_null_fields"},
        )

    # Widen imaging features into *_v{1,2,3}.
    # Exclude scan/acquisition metadata from the analytic merged table.
    excluded_imaging_cols = {"ScanDate", "Protocol"}
    img_feature_cols = [c for c in img.columns if c not in key and str(c) not in excluded_imaging_cols]
    if img_feature_cols:
        wide_parts = []
        for visit in (1, 2, 3):
            sub = img[img["visit"] == visit].copy()
            sub = sub.drop(columns=["visit"], errors="ignore")
            sub = sub.drop(columns=[c for c in sub.columns if str(c) in excluded_imaging_cols], errors="ignore")
            rename = {}
            for c in img_feature_cols:
                if str(c).endswith(f"_v{visit}"):
                    rename[c] = c
                else:
                    rename[c] = f"{c}_v{visit}"
            sub = sub.rename(columns=rename)
            wide_parts.append(sub)

        imaging_wide = wide_parts[0]
        for part in wide_parts[1:]:
            imaging_wide = imaging_wide.merge(part, on="participant_id", how="outer")
    else:
        imaging_wide = img[["participant_id"]].drop_duplicates()

    merged = redcap_wide.merge(
        imaging_wide,
        left_on="ID",
        right_on="participant_id",
        how="left",
        suffixes=("", "_imaging"),
    )
    if "participant_id" in merged.columns:
        merged = merged.drop(columns=["participant_id"])

    audit = pd.DataFrame(audit_rows)
    return merged, audit


def save_trackfa_outputs(
    merged_wide: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    audit: Optional[pd.DataFrame] = None,
    *,
    save_audit: bool = False,
) -> None:
    out_dir = config.trackfa_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_wide.to_csv(config.trackfa_processed_csv, index=False)
    if save_audit and audit is not None:
        audit.to_csv(config.trackfa_audit_csv, index=False)


__all__ = [
    "load_trackfa_redcap_export",
    "collapse_trackfa_events",
    "drop_trackfa_clinical_metadata_columns",
    "select_trackfa_clinical_features",
    "load_trackfa_masterfile",
    "load_trackfa_masterfile_all_sheets",
    "normalize_trackfa_masterfile",
    "merge_trackfa",
    "save_trackfa_outputs",
    "run_merge",
    "qc_long",
    "qc_pairs",
    "imaging_coverage",
    "dry_run_logo_split",
    "delta_summary_table",
]


# ---------------------------------------------------------------------------
# Processed outputs for modeling: long + visit-pairs
# ---------------------------------------------------------------------------


def _strip_trackfa_prefix(participant_id: Any) -> str:
    s = "" if participant_id is None else str(participant_id).strip()
    if s.startswith("TRACKFA_"):
        return s[len("TRACKFA_") :]
    if s.startswith("TRACK-FA_"):
        return s[len("TRACK-FA_") :]
    return s


def _drop_masterfile_duplicates(master_long: pd.DataFrame) -> pd.DataFrame:
    """Drop known duplicated columns created by merging all MasterFile sheets."""
    df = master_long.copy()

    # Exclude acquisition metadata from analytic features.
    df = df.drop(columns=[c for c in ("ScanDate", "Protocol") if c in df.columns], errors="ignore")

    # When `load_trackfa_masterfile_all_sheets` outer-merges sheets, overlapping columns
    # across sheets get suffixed (e.g. *_braindti). Keep the unsuffixed copy.
    dup_suffix = "_braindti"
    dup_cols = [c for c in df.columns if str(c).endswith(dup_suffix)]
    to_drop: list[str] = []
    for c in dup_cols:
        base = str(c)[: -len(dup_suffix)]
        if base in df.columns:
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    return df


def _masterfile_sheet_features(masterfile_xlsx: Path, sheet: str) -> list[str]:
    """Return feature column names for a MasterFile sheet (excluding id/meta columns)."""
    _require_openpyxl()
    preview = pd.read_excel(masterfile_xlsx, sheet_name=sheet, nrows=1)
    drop = {"SubjectID", "Visit", "ScanDate", "Protocol"}
    return [c for c in preview.columns if str(c) not in drop]


def run_merge(
    config: Config = DEFAULT_CONFIG,
    *,
    save: bool = True,
    verbose: bool = True,
    drop_incomplete_clinical_pairs: bool = True,
    require_complete_imaging_sheets: bool = True,
    save_drop3_poms_variant: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end TRACK-FA merge producing `trackfa_long.csv` and `trackfa_pairs.csv`.

    Returns
    -------
    long_df : pd.DataFrame
        522 rows × 161 cols (174 subjects × 3 visits), one row per FRDA subject × visit.
    pairs_df : pd.DataFrame
        One row per consecutive visit pair (V1V2 or V2V3).
    """
    # --- REDCap: collapse events to 1 row per participant -----------------
    redcap = load_trackfa_redcap_export(config.trackfa_redcap_csv)
    redcap_wide, _ = collapse_trackfa_events(redcap)
    redcap_wide = drop_trackfa_clinical_metadata_columns(redcap_wide)
    redcap_wide = select_trackfa_clinical_features(redcap_wide)

    # FRDA only, 174 subjects.
    redcap_wide = redcap_wide[redcap_wide["study_group"] == 0].copy()
    redcap_wide["subject_id"] = redcap_wide["ID"].apply(_strip_trackfa_prefix)

    # --- Imaging: load all sheets, normalize into long (participant_id, visit) ---
    master = load_trackfa_masterfile_all_sheets(config.trackfa_masterfile_xlsx)
    imaging_long, _ = normalize_trackfa_masterfile(master)
    imaging_long = _drop_masterfile_duplicates(imaging_long)

    # Imaging feature list (149).
    imaging_features = [
        c for c in imaging_long.columns if c not in {"participant_id", "visit"}
    ]

    # --- Build long-format analytic table --------------------------------
    demo_cols = ["age", "gender", "gaa_1", "gaa_2", "onset_age", "disease_duration"]
    clinical_base = ["mfars_total", "adl_total", "sara_total"]
    static_cols = [
        "subject_id",
        "visit",
        "site",
        *demo_cols,
        *clinical_base,
        *imaging_features,
    ]

    # Precompute enrolment/static columns per subject.
    base_static = redcap_wide[["subject_id", "site", *demo_cols]].copy()

    visit_frames: list[pd.DataFrame] = []
    for visit in (1, 2, 3):
        vdf = redcap_wide[["subject_id"] + [f"{c}_v{visit}" for c in clinical_base]].copy()
        vdf = vdf.rename(columns={f"{c}_v{visit}": c for c in clinical_base})
        vdf["visit"] = int(visit)

        merged = vdf.merge(base_static, on="subject_id", how="left", validate="one_to_one")

        img = imaging_long[imaging_long["visit"] == visit].copy()
        img["subject_id"] = img["participant_id"].apply(_strip_trackfa_prefix)
        img = img.drop(columns=["participant_id", "visit"], errors="ignore")

        merged = merged.merge(img, on="subject_id", how="left", validate="one_to_one")
        visit_frames.append(merged)

    long_df = pd.concat(visit_frames, ignore_index=True)
    long_df = long_df[static_cols].sort_values(["subject_id", "visit"], kind="stable").reset_index(drop=True)

    # --- Build visit-pair dataset ---------------------------------------
    # Pair features: clinical (3) + imaging (149) = 152.
    pair_features = clinical_base + imaging_features

    # Only form pairs when imaging exists at both visits.
    present = long_df[imaging_features].notna().any(axis=1)
    long_present = long_df.loc[present].copy()
    idx = long_present.set_index(["subject_id", "visit"], drop=False)

    static_for_pairs = long_df[long_df["visit"] == 1].set_index("subject_id")[
        ["site", *demo_cols]
    ]

    pair_rows: list[dict[str, Any]] = []
    for v0, v1, label in ((1, 2, "V1V2"), (2, 3, "V2V3")):
        subs0 = set(long_present[long_present["visit"] == v0]["subject_id"].tolist())
        subs1 = set(long_present[long_present["visit"] == v1]["subject_id"].tolist())
        subs = sorted(subs0 & subs1)

        for sid in subs:
            base = idx.loc[(sid, v0)]
            foll = idx.loc[(sid, v1)]

            static = static_for_pairs.loc[sid].to_dict()
            row: dict[str, Any] = {"subject_id": sid, **static, "pair_type": label}

            base_vals = pd.to_numeric(base[pair_features], errors="coerce")
            foll_vals = pd.to_numeric(foll[pair_features], errors="coerce")
            delta_vals = foll_vals - base_vals

            for f in pair_features:
                row[f"{f}_baseline"] = base_vals.get(f)
                row[f"{f}_followup"] = foll_vals.get(f)
                row[f"delta_{f}"] = delta_vals.get(f)

            pair_rows.append(row)

    pairs_df = pd.DataFrame(pair_rows)

    # Column ordering: 8 static + (152 × 3) features.
    static_pair_cols = ["subject_id", "site", *demo_cols, "pair_type"]
    feature_cols: list[str] = []
    for f in pair_features:
        feature_cols.extend([f"{f}_baseline", f"{f}_followup", f"delta_{f}"])
    pairs_df = pairs_df[static_pair_cols + feature_cols].sort_values(
        ["pair_type", "subject_id"], kind="stable"
    ).reset_index(drop=True)

    if drop_incomplete_clinical_pairs:
        clinical_required = [
            "mfars_total_baseline",
            "mfars_total_followup",
            "delta_mfars_total",
            "adl_total_baseline",
            "adl_total_followup",
            "delta_adl_total",
            "sara_total_baseline",
            "sara_total_followup",
            "delta_sara_total",
        ]
        missing_cols = [c for c in clinical_required if c not in pairs_df.columns]
        if missing_cols:
            raise AssertionError(f"pairs_df missing expected clinical columns: {missing_cols}")

        before = len(pairs_df)
        incomplete = pairs_df[clinical_required].isna().any(axis=1)
        if incomplete.any():
            removed_by_pair = pairs_df.loc[incomplete, "pair_type"].value_counts().to_dict()
            pairs_df = pairs_df.loc[~incomplete].reset_index(drop=True)
            if verbose:
                print(
                    f"[trackfa_pairs] dropped {int(incomplete.sum())} rows with missing clinical values "
                    f"(before={before}, after={len(pairs_df)}). Removed by pair: {removed_by_pair}"
                )

    # ------------------------------------------------------------------
    # Option B: retain more rows by dropping a small number of problematic POMs
    # features, then enforcing strict completeness on the remaining imaging bases.
    #
    # This is saved as an additional CSV and does not change the main output.
    # ------------------------------------------------------------------
    if save and save_drop3_poms_variant:
        drop_bases = {"tNAA_myo_Ins", "DN_suscept", "DN_vol"}
        variant_imaging_bases = [b for b in imaging_features if b not in drop_bases]

        # Drop the columns for these bases (baseline/followup/delta).
        cols_to_drop: list[str] = []
        for b in sorted(drop_bases):
            cols_to_drop.extend([f"{b}_baseline", f"{b}_followup", f"delta_{b}"])

        variant = pairs_df.drop(columns=[c for c in cols_to_drop if c in pairs_df.columns]).copy()

        # Require all remaining imaging bases to be present at baseline and followup.
        req_base = [f"{b}_baseline" for b in variant_imaging_bases]
        req_foll = [f"{b}_followup" for b in variant_imaging_bases]
        missing_req = [c for c in (req_base + req_foll) if c not in variant.columns]
        if missing_req:
            raise AssertionError(
                "Option-B variant missing expected imaging columns. "
                f"Example missing: {missing_req[:10]}"
            )

        keep = variant[req_base].notna().all(axis=1) & variant[req_foll].notna().all(axis=1)
        variant = variant.loc[keep].reset_index(drop=True)

        # Build patient_id and drop identity columns, matching the main pairs output.
        variant.insert(
            0,
            "patient_id",
            variant["subject_id"].astype(str) + "_" + variant["pair_type"].astype(str),
        )
        if variant["patient_id"].duplicated().any():
            raise AssertionError("Option-B variant patient_id is not unique.")
        variant = variant.drop(columns=["subject_id", "pair_type"])

        out_dir = config.trackfa_processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        variant_path = out_dir / "trackfa_pairs_drop3poms.csv"
        variant.to_csv(variant_path, index=False)
        if verbose:
            pair_counts = (
                variant["patient_id"]
                .astype(str)
                .str.split("_", n=1, expand=True)[1]
                .value_counts()
                .to_dict()
            )
            print(
                f"[saved] {variant_path} ({variant.shape[0]} rows × {variant.shape[1]} cols); "
                f"pair counts: {pair_counts}"
            )

    # Enforce strict imaging-sheet completeness per visit (baseline + followup).
    # This removes "empty-heavy" pairs where one whole imaging modality/sheet is missing.
    if require_complete_imaging_sheets:
        sheets = {
            "POMs": _masterfile_sheet_features(config.trackfa_masterfile_xlsx, "POMs"),
            "BrainSpineMorph": _masterfile_sheet_features(
                config.trackfa_masterfile_xlsx, "BrainSpineMorph"
            ),
            "BrainDTI": _masterfile_sheet_features(config.trackfa_masterfile_xlsx, "BrainDTI"),
        }

        sheet_masks: dict[str, pd.Series] = {}
        for sheet_name, feats in sheets.items():
            base_cols = [f"{c}_baseline" for c in feats]
            foll_cols = [f"{c}_followup" for c in feats]

            missing_base = [c for c in base_cols if c not in pairs_df.columns]
            missing_foll = [c for c in foll_cols if c not in pairs_df.columns]
            if missing_base or missing_foll:
                raise AssertionError(
                    f"pairs_df missing expected {sheet_name} columns. "
                    f"missing_baseline={missing_base[:10]} missing_followup={missing_foll[:10]}"
                )

            sheet_masks[f"{sheet_name}_complete_baseline"] = pairs_df[base_cols].notna().all(
                axis=1
            )
            sheet_masks[f"{sheet_name}_complete_followup"] = pairs_df[foll_cols].notna().all(
                axis=1
            )

        keep = pd.Series(True, index=pairs_df.index)
        for m in sheet_masks.values():
            keep &= m

        if (~keep).any():
            removed = pairs_df.loc[~keep].copy()
            removed_by_pair = removed["pair_type"].value_counts().to_dict()

            failed_counts: dict[str, int] = {}
            for name, mask in sheet_masks.items():
                failed_counts[name] = int((~mask & ~keep).sum())

            # Identify "top features driving failure" on removed rows: highest missing fraction
            # among feature columns (baseline/followup/delta).
            static_cols = {"subject_id", "pair_type", "site", *demo_cols}
            feat_cols = [c for c in pairs_df.columns if c not in static_cols]
            miss_removed = removed[feat_cols].isna().mean().sort_values(ascending=False)
            top_missing_features = miss_removed.head(15)

            if verbose:
                print(
                    f"[trackfa_pairs] dropped {int((~keep).sum())} rows failing strict sheet completeness "
                    f"(before={len(pairs_df)}, after={int(keep.sum())}). Removed by pair: {removed_by_pair}"
                )
                failed_counts_str = ", ".join(
                    f"{k}={v}" for k, v in failed_counts.items()
                )
                print(
                    "[trackfa_pairs] failed-sheet counts (rows where this condition fails): "
                    + failed_counts_str
                )
                print("[trackfa_pairs] top missing columns among removed rows:")
                print(top_missing_features.to_string())

            # Save a diagnostic CSV alongside the processed outputs.
            if save:
                miss_report = removed[["subject_id", "pair_type", "site", *demo_cols]].copy()
                for name, mask in sheet_masks.items():
                    miss_report[name] = mask.loc[removed.index].to_numpy()
                miss_report["row_missing_fraction"] = removed.isna().mean(axis=1)
                miss_path = config.trackfa_processed_dir / "trackfa_pairs_miss.csv"
                miss_report.to_csv(miss_path, index=False)
                if verbose:
                    print(f"[saved] {miss_path}")

            pairs_df = pairs_df.loc[keep].reset_index(drop=True)

    # Primary key for downstream modeling: {subject_id}_{pair} (e.g., AAN001_V1V2).
    pairs_df.insert(
        0,
        "patient_id",
        pairs_df["subject_id"].astype(str) + "_" + pairs_df["pair_type"].astype(str),
    )
    if pairs_df["patient_id"].duplicated().any():
        dup = pairs_df.loc[
            pairs_df["patient_id"].duplicated(keep=False),
            ["patient_id", "subject_id", "pair_type"],
        ].head(20)
        raise AssertionError(
            f"patient_id is not unique. Examples:\n{dup.to_string(index=False)}"
        )

    # Drop source columns after merging into patient_id.
    pairs_df = pairs_df.drop(columns=["subject_id", "pair_type"])

    # --- Verification invariants ---------------------------------------
    if long_df.shape != (522, 161):
        raise AssertionError(f"trackfa_long expected 522×161, got {long_df.shape}")
    if long_df["subject_id"].nunique() != 174:
        raise AssertionError(
            f"trackfa_long expected 174 unique subjects, got {long_df['subject_id'].nunique()}"
        )
    if pairs_df.shape[1] != 464:
        raise AssertionError(f"trackfa_pairs expected 464 columns, got {pairs_df.shape[1]}")
    # Pair label lives in patient_id suffix.
    pair_counts = pairs_df["patient_id"].astype(str).str.split("_", n=1, expand=True)[1].value_counts().to_dict()
    if verbose:
        print(f"[trackfa_long] {long_df.shape[0]} rows × {long_df.shape[1]} cols")
        print(f"[trackfa_long] unique subjects: {long_df['subject_id'].nunique()}")
        print(f"[trackfa_pairs] {pairs_df.shape[0]} rows × {pairs_df.shape[1]} cols")
        print(f"[trackfa_pairs] pair counts: {pair_counts}")

    if drop_incomplete_clinical_pairs:
        clinical_required = [
            "mfars_total_baseline",
            "mfars_total_followup",
            "delta_mfars_total",
            "adl_total_baseline",
            "adl_total_followup",
            "delta_adl_total",
            "sara_total_baseline",
            "sara_total_followup",
            "delta_sara_total",
        ]
        if pairs_df[clinical_required].isna().any(axis=1).any():
            raise AssertionError("pairs_df still contains missing clinical values after filtering.")

    if require_complete_imaging_sheets:
        # Sanity: strict completeness implies no NaNs across imaging feature columns.
        non_feature = {
            "patient_id",
            "site",
            *demo_cols,
            "mfars_total_baseline",
            "mfars_total_followup",
            "delta_mfars_total",
            "adl_total_baseline",
            "adl_total_followup",
            "delta_adl_total",
            "sara_total_baseline",
            "sara_total_followup",
            "delta_sara_total",
        }
        feature_cols = [c for c in pairs_df.columns if c not in non_feature]
        if pairs_df[feature_cols].isna().any(axis=1).any():
            raise AssertionError("pairs_df contains missing imaging features despite strict completeness filter.")

    if save:
        out_dir = config.trackfa_processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        long_path = out_dir / "trackfa_long.csv"
        pairs_path = out_dir / "trackfa_pairs.csv"
        long_df.to_csv(long_path, index=False)
        pairs_df.to_csv(pairs_path, index=False)
        if verbose:
            print(f"[saved] {long_path}")
            print(f"[saved] {pairs_path}")

    return long_df, pairs_df


def qc_long(long_df: pd.DataFrame) -> dict[str, Any]:
    demo_cols = ["age", "gender", "gaa_1", "gaa_2", "onset_age", "disease_duration"]
    clinical_base = ["mfars_total", "adl_total", "sara_total"]
    imaging_cols = [
        c
        for c in long_df.columns
        if c
        not in {"subject_id", "visit", "site", *demo_cols, *clinical_base}
    ]

    out: dict[str, Any] = {
        "n_rows": int(len(long_df)),
        "n_cols": int(long_df.shape[1]),
        "n_subjects": int(long_df["subject_id"].nunique()),
        "visit_distribution": long_df["visit"].value_counts().sort_index().to_frame("n"),
        "missingness_by_group": pd.DataFrame(
            {
                "demo": [float(long_df[demo_cols].isna().mean().mean())],
                "clinical": [float(long_df[clinical_base].isna().mean().mean())],
                "imaging": [float(long_df[imaging_cols].isna().mean().mean())],
            }
        ),
    }

    # Heatmap: missing fraction by column, grouped.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        miss = pd.concat(
            [
                long_df[demo_cols].isna().mean().rename("demo"),
                long_df[clinical_base].isna().mean().rename("clinical"),
                long_df[imaging_cols].isna().mean().rename("imaging"),
            ],
            axis=1,
        ).T
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(miss, cmap="viridis", cbar_kws={"label": "missing fraction"}, ax=ax)
        ax.set_title("Missingness heatmap (by column)")
        ax.set_xlabel("column")
        ax.set_ylabel("group")
        fig.tight_layout()
        out["missing_heatmap_fig"] = fig
    except Exception as e:  # pragma: no cover
        out["missing_heatmap_fig"] = None
        out["missing_heatmap_error"] = str(e)

    return out


def qc_pairs(pairs_df: pd.DataFrame) -> dict[str, Any]:
    pair_series = pairs_df["patient_id"].astype(str).str.split("_", n=1, expand=True)[1]
    subject_series = pairs_df["patient_id"].astype(str).str.split("_", n=1, expand=True)[0]
    out: dict[str, Any] = {
        "n_rows": int(len(pairs_df)),
        "n_cols": int(pairs_df.shape[1]),
        "pair_counts": pair_series.value_counts().to_frame("n"),
        "subjects_with_both_pair_types": int(
            (pd.DataFrame({"subject_id": subject_series, "pair": pair_series}).groupby("subject_id")["pair"].nunique() == 2).sum()
        ),
        "subjects_with_one_pair_type": int(
            (pd.DataFrame({"subject_id": subject_series, "pair": pair_series}).groupby("subject_id")["pair"].nunique() == 1).sum()
        ),
    }

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(pairs_df["delta_mfars_total"].dropna(), bins=30, ax=axes[0])
        axes[0].set_title("delta_mfars_total")
        sns.histplot(pairs_df["delta_sara_total"].dropna(), bins=30, ax=axes[1])
        axes[1].set_title("delta_sara_total")
        fig.tight_layout()
        out["delta_hist_fig"] = fig
    except Exception as e:  # pragma: no cover
        out["delta_hist_fig"] = None
        out["delta_hist_error"] = str(e)

    return out


def imaging_coverage(long_df: pd.DataFrame, *, missing_threshold: float = 0.20) -> pd.DataFrame:
    demo_cols = ["age", "gender", "gaa_1", "gaa_2", "onset_age", "disease_duration"]
    clinical_base = ["mfars_total", "adl_total", "sara_total"]
    imaging_cols = [
        c
        for c in long_df.columns
        if c
        not in {"subject_id", "visit", "site", *demo_cols, *clinical_base}
    ]
    frac_non_nan = long_df[imaging_cols].notna().mean().sort_values(ascending=False)
    out = frac_non_nan.to_frame("frac_non_nan")
    out["frac_missing"] = 1.0 - out["frac_non_nan"]
    out["flag_gt20pct_missing"] = out["frac_missing"] > float(missing_threshold)
    return out


def dry_run_logo_split(pairs_df: pd.DataFrame) -> dict[str, Any]:
    """Assert subject-level split integrity under Leave-One-Group-Out."""
    from sklearn.model_selection import LeaveOneGroupOut

    # `patient_id` is `{subject_id}_{pair}`; groups are subject_id.
    groups = pairs_df["patient_id"].astype(str).str.split("_", n=1, expand=True)[0].to_numpy()
    logo = LeaveOneGroupOut()
    X = np.zeros((len(pairs_df), 1))

    for train_idx, test_idx in logo.split(X, groups=groups):
        train_g = set(groups[train_idx])
        test_g = set(groups[test_idx])
        if train_g & test_g:
            raise AssertionError("Subject leakage: subject_id appears in both train and held-out set.")

    return {"ok": True, "n_splits": int(logo.get_n_splits(groups=groups))}


def delta_summary_table(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """One-line per clinical score: mean±std(delta) and SRM = mean/std."""
    clinical_base = ["mfars_total", "adl_total", "sara_total"]
    rows: list[dict[str, Any]] = []
    for score in clinical_base:
        d = pd.to_numeric(pairs_df[f"delta_{score}"], errors="coerce").dropna()
        mean = float(d.mean()) if len(d) else np.nan
        std = float(d.std(ddof=1)) if len(d) > 1 else np.nan
        srm = mean / std if std and not np.isnan(std) else np.nan
        rows.append(
            {
                "clinical_score": score,
                "delta_mean": mean,
                "delta_std": std,
                "SRM_mean_over_std": srm,
                "n": int(len(d)),
            }
        )
    return pd.DataFrame(rows)


def feature_catalog(long_df: pd.DataFrame, pairs_df: pd.DataFrame) -> dict[str, Any]:
    """Return final feature names grouped by category for review.

    Categories are based on the merged outputs produced by `run_merge`.
    """
    demo = ["site", "age", "gender", "gaa_1", "gaa_2", "onset_age", "disease_duration"]
    clinical = ["mfars_total", "adl_total", "sara_total"]

    long_cols = list(long_df.columns)
    long_imaging = [
        c for c in long_cols if c not in {"subject_id", "visit", *demo, *clinical}
    ]

    pair_cols = list(pairs_df.columns)
    pair_static = [c for c in demo if c in pair_cols]
    pair_clinical_cols = [
        c
        for c in pair_cols
        if (
            c.startswith("mfars_total_")
            or c.startswith("adl_total_")
            or c.startswith("sara_total_")
            or c.startswith("delta_mfars_total")
            or c.startswith("delta_adl_total")
            or c.startswith("delta_sara_total")
        )
    ]

    # Base imaging names inferred from *_baseline columns.
    imaging_bases: list[str] = []
    for c in pair_cols:
        if c.endswith("_baseline") and c not in {
            "mfars_total_baseline",
            "adl_total_baseline",
            "sara_total_baseline",
        }:
            imaging_bases.append(c[: -len("_baseline")])
    imaging_bases = sorted(set(imaging_bases))

    return {
        "demographic_columns": demo,
        "clinical_base_names": clinical,
        "long_imaging_columns": sorted(long_imaging),
        "pairs_static_columns": ["patient_id", *pair_static],
        "pairs_clinical_columns": sorted(set(pair_clinical_cols)),
        "pairs_imaging_base_names": imaging_bases,
    }
