import numpy as np
import pandas as pd

from src.eval.metrics import (
    bootstrap_paired_metric,
    compute_longitudinal_deltas,
    paired_cohens_dz,
    probability_positive_change,
)


def test_paired_cohens_dz_uses_within_subject_delta_sd():
    deltas = np.array([1.0, 2.0, 3.0])
    assert paired_cohens_dz(deltas) == np.mean(deltas) / np.std(deltas, ddof=1)
    assert np.isnan(paired_cohens_dz([1.0]))
    assert np.isnan(paired_cohens_dz([2.0, 2.0]))


def test_compute_longitudinal_deltas_observed_pairs_only_and_annualises():
    scores = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "c", "c"],
            "visit": ["V1", "V3", "V1", "V1", "V3"],
            "time_years": [0.0, 2.2, 0.0, 0.0, 2.0],
            "score": [1.0, 3.2, 2.0, 5.0, 6.0],
        }
    )

    out = compute_longitudinal_deltas(scores, "V1", "V3", annualise=True)

    assert out["subject_id"].tolist() == ["a", "c"]
    assert out["delta"].tolist() == [2.2, 1.0]
    assert out["annualised_delta"].round(3).tolist() == [1.0, 0.5]


def test_probability_and_bootstrap_metric_report_denominator():
    deltas = np.array([1.0, -1.0, 2.0, np.nan])
    assert probability_positive_change(deltas) == 2 / 3
    out = bootstrap_paired_metric(deltas, paired_cohens_dz, n_boot=50, seed=7)
    assert set(out) == {"point", "ci_low", "ci_high", "n"}
    assert out["n"] == 3
