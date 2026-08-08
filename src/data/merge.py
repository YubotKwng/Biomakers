"""Merge utilities for legacy Melbourne/Campinas FRDA cohort files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import Config, DEFAULT_CONFIG
from .ids import (
    std_col,
    find_id_column,
    parse_bids_subject_session,
    campac_to_pac,
    pac_to_campac,
)
from .loading import (
    extract_subject_from_name,
    is_ses3,
    load_file_df,
    find_col,
    is_melfrd_clinical_id,
    clinical_id_to_bids,
)


# ---------------------------------------------------------------------------
# Long-format visit preview
# ---------------------------------------------------------------------------
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a legacy wide V1/V2 table into one row per subject visit."""
    # Build the paired feature list from *_v1 and *_v2 columns.
    v1_cols = [c for c in df.columns if c.endswith('_v1')]
    v2_cols = [c for c in df.columns if c.endswith('_v2')]
    base_cols = sorted({c[:-3] for c in v1_cols} & {c[:-3] for c in v2_cols})

    rows = []
    for _, row in df.iterrows():
        subj = row.get('subject', row.get(find_id_column(df)))
        # Visit 1 row.
        r1 = {'subject': subj, 'visit': 1}
        for b in base_cols:
            r1[b] = row.get(b + '_v1')
        r1['FARS'] = row.get('FARS1')
        r1['SARA'] = row.get('SARA1')
        rows.append(r1)
        # Visit 2 row.
        r2 = {'subject': subj, 'visit': 2}
        for b in base_cols:
            r2[b] = row.get(b + '_v2')
        r2['FARS'] = row.get('FARS2')
        r2['SARA'] = row.get('SARA2')
        rows.append(r2)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-cohort merge orchestration
# ---------------------------------------------------------------------------
def merge_cohorts(config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Merge raw Melbourne/Campinas inputs into a wide analysis table.

    The returned frame contains paired visit imaging columns, clinical scores,
    demographic covariates, and an internal ``subject`` column for downstream
    cohort splitting and integrity checks.
    """
    raw_dir = config.raw_data_dir

    # Load raw cohort and imaging files.
    mri_path = raw_dir / "mri_data.csv"
    roi_path = raw_dir / "ROI_vbcb_p50_Vwm.csv"
    demo_melb_path = raw_dir / "IMAGE_FRDA_demograhics.xlsx"
    demo_camp_path = raw_dir / "Campinas_Demographics_FRDA_.xlsx"
    v1v2_path = raw_dir / "raw_frda_v1_v2_updated.xlsx"
    sca_path = raw_dir / "frda_csa_table.xlsx"
    ecc_path = raw_dir / "frda_eccentricity_table.xlsx"

    # Load the wide clinical/imaging master table.
    master = pd.read_excel(v1v2_path)

    id_col = find_id_column(master)
    master_cols_std = {std_col(c): c for c in master.columns}  # noqa: F841

    # Build Campinas session-3 MRI lookup.
    mri_df = pd.read_csv(mri_path)
    mri_df.columns = mri_df.columns.str.strip()

    parsed = mri_df["bidsID"].apply(parse_bids_subject_session)
    mri_df["subject"] = parsed.apply(lambda x: x[0])
    mri_df["session"] = parsed.apply(lambda x: x[1])

    rename = {
        "SCP_FA": "scp_fa", "MCP_FA": "mcp_fa", "ICP_FA": "icp_fa",
        "SCP_MD": "scp_md", "MCP_MD": "mcp_md", "ICP_MD": "icp_md",
        "SCP_AD": "scp_ad", "MCP_AD": "mcp_ad", "ICP_AD": "icp_ad",
        "SCP_RD": "scp_rd", "MCP_RD": "mcp_rd", "ICP_RD": "icp_rd",
        "Medulla_vol": "medulla_vol",
        "Midbrain_vol": "midbrain_vol",
        "Pons_vol": "pons_vol",
        "cb_ant_vol": "cb_ant_vol",
        "cb_sup_post_vol": "cb_sup_post_vol",
        "cb_inf_post_vol": "cb_inf_post_vol",
        "cb_floc_vol": "cb_floc_vol",
        "cb_vermis_vol": "cb_vermis_vol",
    }
    mri_df = mri_df.rename(columns=rename)

    camp_ses3 = mri_df[
        (mri_df["subject"].str.startswith("sub-campac"))
        & (mri_df["session"] == "ses-3")
    ].copy()

    mri_lookup = {}
    for _, row in camp_ses3.iterrows():
        subj = row["subject"]
        feats = {std_col(k): row[k] for k in row.index if k not in ["bidsID", "subject", "session"]}
        mri_lookup[subj] = feats

    # Build Campinas session-3 ROI lookup.
    roi_df = pd.read_csv(roi_path)
    roi_df.columns = roi_df.columns.str.strip()

    parsed = roi_df["names"].apply(parse_bids_subject_session)
    roi_df["subject"] = parsed.apply(lambda x: x[0])
    roi_df["session"] = parsed.apply(lambda x: x[1])

    ROI_SCALE = 1000.0
    roi_df["scp_vol"] = ((roi_df["rSCP"] + roi_df["lSCP"]) / 2.0) * ROI_SCALE
    roi_df["mcp_vol"] = ((roi_df["rMCP"] + roi_df["lMCP"]) / 2.0) * ROI_SCALE
    roi_df["icp_vol"] = ((roi_df["rICP"] + roi_df["lICP"]) / 2.0) * ROI_SCALE

    camp_roi = roi_df[
        (roi_df["subject"].str.startswith("sub-campac"))
        & (roi_df["session"] == "ses-3")
    ].copy()

    roi_lookup = {}
    for _, row in camp_roi.iterrows():
        subj = row["subject"]
        feats = {"scp_vol": row["scp_vol"], "mcp_vol": row["mcp_vol"], "icp_vol": row["icp_vol"]}
        roi_lookup[subj] = {std_col(k): v for k, v in feats.items()}

    # Build Campinas session-3 spinal cord CSA/ECC lookups.
    ecc_df = pd.read_excel(ecc_path)
    ecc_name_col = find_id_column(ecc_df)
    _ecc_ses3 = ecc_df[ecc_df[ecc_name_col].apply(is_ses3)].copy()

    ecc_lookup = {}
    for _, row in _ecc_ses3.iterrows():
        sub_id = extract_subject_from_name(row[ecc_name_col])
        if not sub_id:
            continue
        ecc_lookup[sub_id] = {
            "ECC_C1_v2": row.get("Mean(eccentricity) C1", np.nan),
            "ECC_C2_v2": row.get("Mean(eccentricity) C2", np.nan),
        }

    csa_df = pd.read_excel(sca_path)
    csa_name_col = find_id_column(csa_df)
    _csa_ses3 = csa_df[csa_df[csa_name_col].apply(is_ses3)].copy()

    csa_lookup = {}
    for _, row in _csa_ses3.iterrows():
        sub_id = extract_subject_from_name(row[csa_name_col])
        if not sub_id:
            continue
        csa_lookup[sub_id] = {
            "CSA_C1_v2": row.get("Mean(CSA) C1", np.nan),
            "CSA_C2_v2": row.get("Mean(CSA) C2", np.nan),
        }

    # Map Campinas session-3 source columns into visit-2 feature columns.
    MAPPING = {
        "CSA_C2_v2": ("frda_csa_table.xlsx", "CSA C2"),
        "CSA_C1_v2": ("frda_csa_table.xlsx", "CSA C2"),
        "ECC_C2_v2": ("frda_eccentricity_table.xlsx", "Mean(eccentricity) C2"),
        "ECC_C1_v2": ("frda_eccentricity_table.xlsx", "Mean(eccentricity) C1"),
        "SCP_v2": ("ROI_vbcb_p50_Vwm.csv", "scp_vol"),
        "MCP_v2": ("ROI_vbcb_p50_Vwm.csv", "mcp_vol"),
        "ICP_v2": ("ROI_vbcb_p50_Vwm.csv", "icp_vol"),
        "RDICP_v2": ("mri_data.csv", "ICP_RD"),
        "RDMCP_v2": ("mri_data.csv", "MCP_RD"),
        "RDSCP_v2": ("mri_data.csv", "SCP_RD"),
        "ADICP_v2": ("mri_data.csv", "ICP_AD"),
        "ADMCP_v2": ("mri_data.csv", "MCP_AD"),
        "ADSCP_v2": ("mri_data.csv", "SCP_AD"),
        "MDICP_v2": ("mri_data.csv", "ICP_MD"),
        "MDMCP_v2": ("mri_data.csv", "MCP_MD"),
        "MDSCP_v2": ("mri_data.csv", "SCP_MD"),
        "FAICP_v2": ("mri_data.csv", "ICP_FA"),
        "FAMCP_v2": ("mri_data.csv", "MCP_FA"),
        "FASCP_v2": ("mri_data.csv", "SCP_FA"),
        "Midbrain_v2": ("mri_data.csv", "Midbrain_vol"),
        "Medulla_v2": ("mri_data.csv", "Medulla_vol"),
        "Pons_v2": ("mri_data.csv", "Pons_vol"),
        "VermisCBLM_v2": ("mri_data.csv", "cb_vermis_vol"),
        "FlocCBLM_v2": ("mri_data.csv", "cb_floc_vol"),
        "InfPostCBLM_v2": ("mri_data.csv", "cb_inf_post_vol"),
        "SupPostCBLM_v2": ("mri_data.csv", "cb_sup_post_vol"),
        "AntCBLM_v2": ("mri_data.csv", "cb_ant_vol"),
    }

    id_col_master = find_id_column(master)

    # Cache raw file dataframes.
    file_cache = {}
    for raw_col, (fname, _) in MAPPING.items():
        if fname not in file_cache:
            file_cache[fname] = load_file_df(fname, raw_dir)

    explicit_lookup = {}
    for raw_col, (fname, src_col) in MAPPING.items():
        df = file_cache[fname]
        id_col = find_id_column(df)
        ses3 = df[df[id_col].astype(str).str.contains('ses-3', na=False)].copy()

        for _, row in ses3.iterrows():
            sub_id = None
            m = re.search(r"(sub-campac\d+)", str(row[id_col]))
            if m:
                sub_id = m.group(1)
            if not sub_id:
                continue
            if sub_id not in explicit_lookup:
                explicit_lookup[sub_id] = {}
            explicit_lookup[sub_id][raw_col] = row.get(src_col, np.nan)

    # Fill only missing visit-2 values in the master table.
    filled = 0
    for idx, row in master.iterrows():
        sid = str(row[id_col_master])
        if "campac" not in sid and "pac" not in sid:
            continue
        sub_id = sid if "sub-campac" in sid else pac_to_campac(sid)
        if not sub_id or sub_id not in explicit_lookup:
            continue
        for raw_col, value in explicit_lookup[sub_id].items():
            if raw_col in master.columns and pd.isna(row[raw_col]):
                master.at[idx, raw_col] = value
                filled += 1

    # Load Melbourne and Campinas demographic covariates.
    melb = pd.read_excel(demo_melb_path, sheet_name="ALL DATA MERGED", header=None)
    header = melb.iloc[2].astype(str).str.strip()
    melb_data = melb.iloc[3:].copy()
    melb_data.columns = range(melb_data.shape[1])
    melb_data = melb_data.reset_index(drop=True)

    header_lc = header.str.lower()
    name_to_idx = {header_lc[i]: i for i in range(len(header_lc))}

    COL_CLINICAL_ID = find_col(name_to_idx, 'clinical id', 'clinical_id', 'id') or 0
    COL_AGE1 = find_col(name_to_idx, 'age1', 'age 1')
    COL_AGE2 = find_col(name_to_idx, 'age2', 'age 2')
    COL_FARS1 = find_col(name_to_idx, 'fars', 'fars1')
    COL_FARS2 = find_col(name_to_idx, 'fars2', 'fars 2')
    COL_SARA1 = find_col(name_to_idx, 'sara1', 'sara 1')
    COL_SARA2 = find_col(name_to_idx, 'sara2', 'sara 2')
    COL_AGE_ONSET = find_col(name_to_idx, 'age onset', 'age at onset', 'aao')
    COL_DIS_DUR1 = find_col(name_to_idx, 'disease dur', 'disease duration', 'dur1', 'disease dur1')
    COL_DIS_DUR2 = find_col(name_to_idx, 'disease dur2', 'dur2', 'disease duration 2')
    COL_GAA1 = find_col(name_to_idx, 'gaa1')
    COL_GAA2 = find_col(name_to_idx, 'gaa2')

    sex_candidates = header_lc[header_lc.str.contains('sex|gender', regex=True, na=False)].index.tolist()
    COL_SEX = sex_candidates[0] if sex_candidates else None

    melb_subset = melb_data[melb_data[COL_CLINICAL_ID].apply(is_melfrd_clinical_id)].copy()

    melb_rows = []
    for _, row in melb_subset.iterrows():
        try:
            cid = float(row[COL_CLINICAL_ID])
        except Exception:
            continue
        bids_id = clinical_id_to_bids(cid)

        def safe_float(x):
            """Coerce messy spreadsheet cells to float or NaN."""
            try:
                v = float(x)
                return v if not np.isnan(v) else np.nan
            except Exception:
                return np.nan

        melb_rows.append({
            'subject': bids_id,
            'sex': safe_float(row[COL_SEX]) if COL_SEX is not None else np.nan,
            'age1': safe_float(row[COL_AGE1]) if COL_AGE1 is not None else np.nan,
            'age2': safe_float(row[COL_AGE2]) if COL_AGE2 is not None else np.nan,
            'FARS1': safe_float(row[COL_FARS1]) if COL_FARS1 is not None else np.nan,
            'FARS2': safe_float(row[COL_FARS2]) if COL_FARS2 is not None else np.nan,
            'SARA1': safe_float(row[COL_SARA1]) if COL_SARA1 is not None else np.nan,
            'SARA2': safe_float(row[COL_SARA2]) if COL_SARA2 is not None else np.nan,
            'age_at_onset': safe_float(row[COL_AGE_ONSET]) if COL_AGE_ONSET is not None else np.nan,
            'dur1': safe_float(row[COL_DIS_DUR1]) if COL_DIS_DUR1 is not None else np.nan,
            'dur2': safe_float(row[COL_DIS_DUR2]) if COL_DIS_DUR2 is not None else np.nan,
            'GAA1': safe_float(row[COL_GAA1]) if COL_GAA1 is not None else np.nan,
            'GAA2': safe_float(row[COL_GAA2]) if COL_GAA2 is not None else np.nan,
        })

    melb_df = pd.DataFrame(melb_rows)

    camp = pd.read_excel(demo_camp_path)
    camp_id_col = find_id_column(camp)

    camp_rows = []
    for _, row in camp.iterrows():
        pac_id = row[camp_id_col]
        sub_id = pac_to_campac(pac_id)
        if not sub_id:
            continue
        camp_rows.append({
            'subject': sub_id,
            **{std_col(k): row[k] for k in camp.columns if k != camp_id_col}
        })

    camp_df = pd.DataFrame(camp_rows)

    # Merge demographic covariates into the master table.
    id_col_master = find_id_column(master)
    master_subject = []
    for sid in master[id_col_master].astype(str):
        if "sub-melfrd" in sid or "sub-campac" in sid:
            master_subject.append(sid)
        elif "pac" in sid:
            master_subject.append(pac_to_campac(sid))
        else:
            master_subject.append(sid)
    master["subject"] = master_subject

    master = master.merge(melb_df, on="subject", how="left", suffixes=("", "_melb"))
    master = master.merge(camp_df, on="subject", how="left", suffixes=("", "_camp"))

    # Stash the v1v2 raw column list on the master so callers can reuse
    # it without re-reading the Excel header.
    master.attrs["v1v2_path"] = str(v1v2_path)

    return master


# ---------------------------------------------------------------------------
# Split, clean, and write cohort CSVs
# ---------------------------------------------------------------------------
def split_and_save_cohorts(
    master: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build cleaned Melbourne/Campinas cohort frames and write them to disk.

    Returns ``(melb, camp)`` after cohort-specific empty-column pruning. Saves
    four CSVs:

    * ``melbourne_frda_merged.csv``
    * ``campinas_frda_merged.csv``
    * ``combined_frda_merged.csv``
    * ``updated_v1_v2.csv``
    """
    processed_dir = config.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    v1v2_path = Path(master.attrs.get("v1v2_path",
                                      config.raw_data_dir / "raw_frda_v1_v2_updated.xlsx"))

    id_col_master = find_id_column(master)

    # Split rows by cohort identifier.
    melb = master[master[id_col_master].astype(str).str.contains('melfrd', na=False)].copy()
    camp = master[master[id_col_master].astype(str).str.contains('campac|pac', na=False)].copy()

    # Remove internal subject columns before saving public outputs.
    for col in list(melb.columns):
        if col.lower() == 'subject':
            melb = melb.drop(columns=[col])
    for col in list(camp.columns):
        if col.lower() == 'subject':
            camp = camp.drop(columns=[col])

    # Align cohort outputs to the union of available columns.
    all_cols = sorted(set(melb.columns) | set(camp.columns))
    for col in all_cols:
        if col not in melb.columns:
            melb[col] = np.nan
        if col not in camp.columns:
            camp[col] = np.nan
    melb = melb[all_cols]
    camp = camp[all_cols]

    # Drop all-empty columns within each cohort.
    melb = melb.dropna(axis=1, how='all')
    camp = camp.dropna(axis=1, how='all')

    # Order raw visit columns, demographics, clinical scores, then remaining fields.
    raw_cols = list(pd.read_excel(v1v2_path, nrows=0).columns)
    raw_cols = [c for c in raw_cols if c in melb.columns]

    demo_cols = [c for c in ['sex', 'age1', 'age2', 'age_at_onset', 'dur1', 'dur2', 'GAA1', 'GAA2'] if c in melb.columns]
    score_cols = [c for c in ['SARA1', 'SARA2', 'FARA1', 'FARA2', 'FARS1', 'FARS2'] if c in melb.columns]
    remaining = [c for c in melb.columns if c not in raw_cols + demo_cols + score_cols]

    melb = melb[[c for c in raw_cols + demo_cols + score_cols + remaining if c in melb.columns]]
    camp = camp[[c for c in raw_cols + demo_cols + score_cols + remaining if c in camp.columns]]

    # Move cohort ID to the first column.
    if 'ID' in melb.columns:
        cols = ['ID'] + [c for c in melb.columns if c != 'ID']
        melb = melb[cols]
    if 'ID' in camp.columns:
        cols = ['ID'] + [c for c in camp.columns if c != 'ID']
        camp = camp[cols]

    # Write cohort files
    melb.to_csv(processed_dir / 'melbourne_frda_merged.csv', index=False)
    camp.to_csv(processed_dir / 'campinas_frda_merged.csv', index=False)

    # Combined merged (with demographics) — drop all-empty columns and subject column
    combined = master.copy()
    if 'subject' in combined.columns:
        combined = combined.drop(columns=['subject'])
    combined = combined.dropna(axis=1, how='all')
    combined.to_csv(processed_dir / 'combined_frda_merged.csv', index=False)

    # Updated v1/v2 only (same columns as raw_frda_v1_v2_updated.xlsx)
    raw_cols_full = list(pd.read_excel(v1v2_path, nrows=0).columns)
    updated_v1_v2 = master.copy()
    updated_v1_v2 = updated_v1_v2[[c for c in raw_cols_full if c in updated_v1_v2.columns]].copy()
    updated_v1_v2.to_csv(processed_dir / 'updated_v1_v2.csv', index=False)

    return melb, camp
