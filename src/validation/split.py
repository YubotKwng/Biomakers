"""Subject-level split utilities."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.eval.cv import group_kfold_indices


def assert_group_disjoint(train_groups, test_groups) -> None:
    """Raise if any subject/group appears in both train and test folds."""
    train_set = set(np.asarray(train_groups).tolist())
    test_set = set(np.asarray(test_groups).tolist())
    overlap = train_set & test_set
    if overlap:
        examples = sorted(map(str, overlap))[:10]
        raise AssertionError(f"Train/test groups overlap: {examples}")


def repeated_group_kfold_indices(
    groups,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_seed: int = 42,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield repeated group-disjoint folds as ``repeat, fold, train, test``."""
    groups_arr = np.asarray(groups)
    for repeat in range(int(n_repeats)):
        for fold, (train_idx, test_idx) in enumerate(
            group_kfold_indices(groups_arr, n_splits=n_splits, seed=random_seed + repeat),
            start=1,
        ):
            assert_group_disjoint(groups_arr[train_idx], groups_arr[test_idx])
            yield repeat + 1, fold, train_idx, test_idx


__all__ = ["assert_group_disjoint", "repeated_group_kfold_indices"]
