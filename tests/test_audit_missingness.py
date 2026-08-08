import numpy as np
import pandas as pd

from src.data.audit import (
    VISIT_TIME,
    add_visit_time,
    analysis_population_counts,
    audit_visit_patterns,
    visit_pattern_table,
)
from src.data.missingness import (
    feature_missingness_report,
    feature_visit_site_missingness_matrix,
    followup_missingness_analysis,
)


def test_audit_visit_patterns_counts_all_requested_patterns():
    df = pd.DataFrame(
        {
            "subject_id": ["a", "a", "a", "b", "b", "c", "c", "d", "d", "e", "f", "g"],
            "visit": [1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3],
        }
    )

    out = audit_visit_patterns(df, "subject_id", "visit")

    assert out["n_subjects"] == 7
    assert out["patterns"] == {
        "111": 1,
        "110": 1,
        "101": 1,
        "011": 1,
        "100": 1,
        "010": 1,
        "001": 1,
    }
    assert out["n_v1_v2"] == 2
    assert out["n_v2_v3"] == 2
    assert out["n_v1_v3"] == 2
    assert out["n_complete"] == 1

    patterns = visit_pattern_table(out)
    assert patterns.loc[patterns["pattern"] == "111", "meaning"].iloc[0] == "V1, V2, and V3 available"

    populations = analysis_population_counts(df, "subject_id", "visit")
    assert "V1-V3 primary cohort" in populations["population"].tolist()


def test_add_visit_time_uses_canonical_schedule():
    df = pd.DataFrame({"visit": [1, "V2", 3]})
    out = add_visit_time(df)
    assert VISIT_TIME == {"V1": 0.0, "V2": 1.0, "V3": 2.0}
    assert out["time_years"].tolist() == [0.0, 1.0, 2.0]


def test_feature_missingness_report_flags_concentrated_missingness():
    df = pd.DataFrame(
        {
            "visit": ["V1", "V1", "V2", "V2"],
            "site": ["A", "B", "A", "B"],
            "x": [1.0, 2.0, np.nan, np.nan],
            "y": [1.0, np.nan, 3.0, np.nan],
        }
    )

    report = feature_missingness_report(df, ["x", "y", "missing"], concentration_spread=0.4)

    assert set(report) == {"global", "by_visit", "by_site", "by_visit_site", "flags", "missing_features"}
    assert report["global"].set_index("feature").loc["x", "missing_pct"] == 0.5
    assert "x" in set(report["flags"]["feature"])
    assert report["missing_features"]["feature"].tolist() == ["missing"]


def test_feature_visit_site_missingness_matrix_reports_full_strata():
    df = pd.DataFrame(
        {
            "visit": [1, 1, 2, 2],
            "site": ["A", "B", "A", "B"],
            "x": [1.0, np.nan, np.nan, 4.0],
            "y": [np.nan, np.nan, 2.0, 3.0],
        }
    )

    out = feature_visit_site_missingness_matrix(df, ["x", "y"], visit_col="visit", site_col="site")

    assert out["matrix"].loc["x", "V1|A"] == 0.0
    assert out["matrix"].loc["x", "V1|B"] == 1.0
    assert out["matrix"].loc["y", "V2|A"] == 0.0


def test_followup_missingness_analysis_compares_v1_subjects():
    df = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "c", "c"],
            "visit": [1, 3, 1, 1, 3],
            "mfars_total": [10.0, 12.0, 20.0, 30.0, 31.0],
            "site": ["A", "A", "B", "A", "A"],
        }
    )

    out = followup_missingness_analysis(
        df,
        subject_col="subject_id",
        variables=("mfars_total", "site"),
    )

    assert out["n_baseline_subjects"] == 3
    assert out["n_complete_v1_v3"] == 2
    assert out["n_missing_v3"] == 1
    assert {"mfars_total", "site"} <= set(out["comparison"]["variable"])
