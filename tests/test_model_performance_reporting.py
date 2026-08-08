import pandas as pd

from src.reporting.model_performance import (
    PERFORMANCE_SPEC,
    append_log_model_summaries,
    assemble_performance_rows,
)


def test_assemble_performance_rows_uses_required_question_schema():
    composite = pd.DataFrame(
        {
            "interval": ["V1->V3", "V1->V2", "V2->V3"],
            "n_pairs": [10, 10, 9],
            "d_z": [1.2, 0.9, 0.8],
            "d_z_ci_low": [0.7, 0.4, 0.3],
            "d_z_ci_high": [1.6, 1.3, 1.2],
            "p_delta_positive": [0.9, 0.8, 0.78],
        }
    )

    out = assemble_performance_rows("model", composite_intervals=composite)

    assert out.shape[0] == PERFORMANCE_SPEC.shape[0]
    assert out["question"].tolist() == PERFORMANCE_SPEC["question"].tolist()
    assert out.loc[out["question"] == "2-year disease sensitivity", "value"].iloc[0] == 1.2


def test_append_log_model_summaries_keeps_same_schema_for_legacy_models():
    base = assemble_performance_rows("SRM", composite_intervals=pd.DataFrame())
    logs = pd.DataFrame(
        {
            "model": ["Legacy"],
            "d_score": [0.5],
            "d_ci_low": [0.1],
            "d_ci_high": [0.9],
            "n_subjects": [12],
            "cv_mode": ["loo"],
            "source_log": ["log.csv"],
            "notes": ["existing artifact"],
        }
    )

    out = append_log_model_summaries(base, logs)
    legacy = out[out["model"] == "Legacy"]

    assert legacy.shape[0] == PERFORMANCE_SPEC.shape[0]
    assert set(legacy["question"]) == set(PERFORMANCE_SPEC["question"])
