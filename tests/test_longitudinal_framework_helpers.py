import numpy as np
import pandas as pd

from src.data.mri_qc import harmonisation_leakage_policy, mri_outlier_table, site_effect_screen
from src.eval.intervals import interval_effect_summary
from src.eval.model_selection import select_hierarchical_candidate
from src.eval.single_feature import single_feature_interval_baselines
from src.eval.stability import (
    coefficient_sign_stability,
    score_ranking_stability,
    selected_feature_jaccard,
)
from src.models.srm_global import srm_global_loocv, srm_global_repeated_group_cv


def test_interval_summary_keeps_primary_and_secondary_separate():
    scores = pd.DataFrame(
        {
            "subject_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "visit": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "time_years": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "score": [0, 1, 3, 0, 2, 3, 0, 1, 2],
        }
    )

    out = interval_effect_summary(scores, n_boot=50, seed=1)

    assert out["interval"].tolist() == ["V1->V3", "V1->V2", "V2->V3"]
    assert out.set_index("interval").loc["V1->V3", "n_pairs"] == 3
    assert out.set_index("interval").loc["V1->V3", "annualised"]


def test_single_feature_baselines_return_feature_interval_rows():
    df = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b"],
            "visit": [1, 3, 1, 3],
            "time_years": [0, 2, 0, 2],
            "mri_1": [0.0, 2.0, 1.0, 4.0],
        }
    )

    out = single_feature_interval_baselines(df, ["mri_1"], n_boot=20, seed=2)

    assert {"feature", "interval", "d_z", "p_delta_positive"} <= set(out.columns)
    assert "V1->V3" in set(out["interval"])


def test_stability_diagnostics_cover_jaccard_sign_and_ranking():
    jac = selected_feature_jaccard([["a", "b"], ["b", "c"]])
    sign = coefficient_sign_stability([[1, -1], [2, -3], [-1, -2]], ["x", "y"])
    rank = score_ranking_stability(pd.DataFrame({"m1": [1, 2, 3], "m2": [1, 3, 2], "m3": [1, 2, 4]}))

    assert jac["mean_jaccard"] == 1 / 3
    assert sign.set_index("feature").loc["y", "dominant_sign"] == -1
    assert rank["n_pairs"] == 3


def test_hierarchical_candidate_uses_one_se_and_stability_tiebreakers():
    results = pd.DataFrame(
        {
            "name": ["best_unstable", "simple_stable"],
            "mean_validation_dz": [1.0, 0.93],
            "se_validation_dz": [0.1, 0.02],
            "feature_count": [12, 4],
            "jaccard_stability": [0.2, 0.9],
            "p_progression": [0.7, 0.85],
            "score_ranking_stability": [0.4, 0.8],
        }
    )

    chosen = select_hierarchical_candidate(results)

    assert chosen["name"] == "simple_stable"


def test_mri_qc_site_screen_and_policy_tables():
    df = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B", "A", "B"],
            "age": [10, 11, 10, 11, 12, 12],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "mfars_total": [20, 22, 21, 24, 23, 25],
            "mri_1": [1.0, 1.1, 3.0, 3.1, 1.2, 3.2],
        }
    )

    site = site_effect_screen(df, ["mri_1"])
    outliers = mri_outlier_table(df, ["mri_1"], group_cols=("site",))
    policy = harmonisation_leakage_policy()

    assert site.loc[0, "feature"] == "mri_1"
    assert "site_p_value" in site.columns
    assert isinstance(outliers, pd.DataFrame)
    assert policy["rule"].str.contains("training fold", case=False).any()


def test_srm_global_can_target_v1_to_v3_interval():
    df = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "visit": [1, 3, 1, 3, 1, 3, 1, 3],
            "x": [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
            "y": [4.0, 3.0, 4.0, 2.0, 4.0, 1.0, 4.0, 0.0],
        }
    )

    out = srm_global_loocv(
        df,
        ["x", "y"],
        subject_col="subject_id",
        visit_col="visit",
        cv_n_splits=2,
        start_visit=1,
        end_visit=3,
        compute_ci=False,
    )

    assert out["start_visit"] == 1
    assert out["end_visit"] == 3
    assert out["n_subjects"] == 4

    repeated = srm_global_repeated_group_cv(
        df,
        ["x", "y"],
        subject_col="subject_id",
        visit_col="visit",
        n_splits=2,
        n_repeats=2,
        start_visit=1,
        end_visit=3,
        compute_ci=False,
    )
    assert repeated["summary_df"].shape[0] == 2
    assert repeated["n_repeats"] == 2
