"""Experiment reporting utilities."""

from .experiment import save_experiment_artifacts
from .experiment_artifacts import (
    ArtifactValidationError,
    build_experiment_manifest,
    make_participant_folds,
    read_experiment_contract,
    read_table_artifact,
    split_participants_for_fold,
    write_experiment_contract,
    write_table_artifact,
)
from .fold_comparison import fold_train_test_clinical_benchmark_table
from .model_performance import (
    PERFORMANCE_SPEC,
    append_log_model_summaries,
    assemble_performance_rows,
    best_model_rows_from_logs,
    cv_contract_table,
    validation_test_gap_table,
)
from .tables import final_model_performance_matrix

__all__ = [
    "PERFORMANCE_SPEC",
    "ArtifactValidationError",
    "append_log_model_summaries",
    "assemble_performance_rows",
    "best_model_rows_from_logs",
    "build_experiment_manifest",
    "cv_contract_table",
    "final_model_performance_matrix",
    "fold_train_test_clinical_benchmark_table",
    "make_participant_folds",
    "read_experiment_contract",
    "read_table_artifact",
    "save_experiment_artifacts",
    "split_participants_for_fold",
    "validation_test_gap_table",
    "write_experiment_contract",
    "write_table_artifact",
]
