import numpy as np
import pandas as pd
import pytest

from src.data.model_safety import assert_training_frame_is_patient_only
from src.eval.cv import group_kfold_indices
from src.features.selection import select_features, select_one_se_candidate
from src.preprocessing import FeatureMedianImputer, TrainOnlyStandardScaler
from src.validation import assert_group_disjoint


def test_group_kfold_keeps_subjects_disjoint():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    for train_idx, test_idx in group_kfold_indices(groups, n_splits=4, seed=1):
        assert_group_disjoint(groups[train_idx], groups[test_idx])


def test_scaler_and_imputer_fit_only_on_training_rows():
    X_train = np.array([[1.0, np.nan], [3.0, 5.0]])
    X_test = np.array([[100.0, np.nan]])

    imputer = FeatureMedianImputer().fit(X_train)
    scaler = TrainOnlyStandardScaler().fit(imputer.transform(X_train))
    Xt = scaler.transform(imputer.transform(X_test))

    assert imputer.fit_rows_ == 2
    assert scaler.fit_rows_ == 2
    assert Xt.shape == (1, 2)
    assert imputer.statistics_.tolist() == [2.0, 5.0]


def test_feature_selector_uses_caller_supplied_training_matrix_only():
    X_train = pd.DataFrame({"x1": [0, 0, 1, 1], "x2": [10, 11, 12, 13]})
    y_train = np.array([0, 0, 1, 1])
    selected = select_features("mi", X_train, y_train, ["x1", "x2"], k=1)
    assert selected == ["x1"]


def test_model_safety_rejects_clinical_features_and_controls():
    df = pd.DataFrame({"subject": ["a", "b"], "study_group": [0, 0], "FARS": [1, 2], "MRI": [3, 4]})
    with pytest.raises(ValueError, match="Clinical-score"):
        assert_training_frame_is_patient_only(df, ["FARS"])

    controls = pd.DataFrame({"subject": ["a"], "group": ["control"], "MRI": [1.0]})
    with pytest.raises(ValueError, match="Control rows"):
        assert_training_frame_is_patient_only(controls, ["MRI"])


def test_one_se_rule_prefers_simpler_stable_candidate():
    results = pd.DataFrame(
        {
            "name": ["complex_best", "simple_stable", "weak"],
            "mean_validation_dz": [1.00, 0.94, 0.70],
            "se_validation_dz": [0.10, 0.05, 0.02],
            "feature_count": [12, 4, 2],
            "jaccard_stability": [0.5, 0.8, 0.9],
            "sign_stability": [0.6, 0.8, 0.9],
            "directional_consistency": [0.7, 0.8, 0.9],
        }
    )

    chosen = select_one_se_candidate(results)
    assert chosen["name"] == "simple_stable"
