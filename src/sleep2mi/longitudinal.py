"""Longitudinal feedback-score utilities used by the Sleep2MI evaluation.

The functions in this module operate on four-class membership outputs and
participant-level arrays supplied by the caller. They do not load participant
data, fit an encoder, or define a study cohort.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


TASK_VALID_CLASSES: Mapping[str, tuple[int, ...]] = {
    "LR": (1, 2),
    "UD": (3, 4),
    "2D": (1, 2, 3, 4),
}


@dataclass(frozen=True)
class FeedbackAggregation:
    """Equal-weight trial and seed summaries.

    ``seed_scores`` is keyed by ``(participant, task, seed)`` and
    ``ensemble_scores`` by ``(participant, task)``.
    """

    seed_scores: Mapping[tuple[Hashable, str, Hashable], float]
    ensemble_scores: Mapping[tuple[Hashable, str], float]


@dataclass(frozen=True)
class FoldTransform:
    """One outer-fold empirical-midrank transformation."""

    train: np.ndarray
    test: np.ndarray
    imputation_median: float
    midrank_mean: float
    midrank_scale: float


@dataclass(frozen=True)
class OOFResult:
    """Averaged participant-out predictions and evaluation metrics."""

    prediction_full: np.ndarray
    prediction_reduced: np.ndarray
    prediction_intercept: np.ndarray
    test_counts: np.ndarray
    metrics: Mapping[str, float]


def task_valid_membership(
    memberships: np.ndarray,
    target_classes: Sequence[int] | np.ndarray,
    tasks: Sequence[str] | np.ndarray,
    *,
    simplex_tolerance: float = 1e-6,
) -> np.ndarray:
    """Return instructed-target membership within each task-valid class set.

    Class labels are one-based: LR uses classes 1/2, UD uses 3/4, and 2D uses
    all four classes. For row ``i``, the released score is

    ``membership[i, target[i] - 1] / membership[i, valid(task[i])].sum()``.
    """

    values = np.asarray(memberships, dtype=np.float64)
    targets = np.asarray(target_classes)
    task_array = np.asarray(tasks, dtype=object)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("memberships must have shape (observations, 4)")
    if targets.ndim != 1 or task_array.ndim != 1:
        raise ValueError("target_classes and tasks must be one-dimensional")
    if len(targets) != len(values) or len(task_array) != len(values):
        raise ValueError("memberships, target_classes, and tasks must align")
    if not np.all(np.isfinite(values)):
        raise ValueError("memberships must be finite")
    if np.any(values < 0):
        raise ValueError("memberships must be nonnegative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=simplex_tolerance, rtol=0.0):
        raise ValueError("each four-class membership row must sum to one")
    if not np.issubdtype(targets.dtype, np.integer):
        if not np.all(np.equal(targets, np.floor(targets.astype(float)))):
            raise TypeError("target classes must be one-based integers")
        targets = targets.astype(np.int64)
    else:
        targets = targets.astype(np.int64, copy=False)

    result = np.empty(len(values), dtype=np.float64)
    for task in np.unique(task_array):
        task_name = str(task)
        if task_name not in TASK_VALID_CLASSES:
            raise ValueError(f"unknown task: {task_name}")
        rows = np.flatnonzero(task_array == task)
        valid = TASK_VALID_CLASSES[task_name]
        valid_zero_based = np.asarray(valid, dtype=np.int64) - 1
        row_targets = targets[rows]
        if np.any(~np.isin(row_targets, valid)):
            raise ValueError(f"target class is invalid for task {task_name}")
        denominator = values[np.ix_(rows, valid_zero_based)].sum(axis=1)
        if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
            raise ValueError(f"nonpositive task-valid denominator for task {task_name}")
        numerator = values[rows, row_targets - 1]
        result[rows] = numerator / denominator
    return result


def aggregate_trial_and_seed_equal(
    scores: Sequence[float] | np.ndarray,
    participant_ids: Sequence[Hashable],
    tasks: Sequence[str],
    seeds: Sequence[Hashable],
    trial_ids: Sequence[Hashable],
) -> FeedbackAggregation:
    """Average windows within trials, trials within seeds, and seeds equally."""

    values = np.asarray(scores, dtype=np.float64)
    arrays = [participant_ids, tasks, seeds, trial_ids]
    if values.ndim != 1 or any(len(array) != len(values) for array in arrays):
        raise ValueError("all aggregation inputs must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")

    window_groups: dict[tuple[Hashable, str, Hashable, Hashable], list[float]] = (
        defaultdict(list)
    )
    for score, participant, task, seed, trial in zip(
        values, participant_ids, tasks, seeds, trial_ids, strict=True
    ):
        task_name = str(task)
        if task_name not in TASK_VALID_CLASSES:
            raise ValueError(f"unknown task: {task_name}")
        window_groups[(participant, task_name, seed, trial)].append(float(score))

    trial_groups: dict[tuple[Hashable, str, Hashable], list[float]] = defaultdict(list)
    for (participant, task, seed, _trial), window_values in window_groups.items():
        trial_groups[(participant, task, seed)].append(float(np.mean(window_values)))

    seed_scores = {
        key: float(np.mean(trial_values)) for key, trial_values in trial_groups.items()
    }
    participant_groups: dict[tuple[Hashable, str], list[float]] = defaultdict(list)
    for (participant, task, _seed), seed_score in seed_scores.items():
        participant_groups[(participant, task)].append(seed_score)
    ensemble_scores = {
        key: float(np.mean(seed_values))
        for key, seed_values in participant_groups.items()
    }
    return FeedbackAggregation(seed_scores=seed_scores, ensemble_scores=ensemble_scores)


def outer_training_midrank(
    train_scores: Sequence[float] | np.ndarray,
    test_scores: Sequence[float] | np.ndarray,
) -> FoldTransform:
    """Impute, empirical-midrank, and standardize using outer training only."""

    train = np.asarray(train_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    if train.ndim != 1 or test.ndim != 1 or len(train) == 0:
        raise ValueError("train_scores and test_scores must be one-dimensional")
    finite_train = train[np.isfinite(train)]
    if len(finite_train) == 0:
        raise ValueError("outer training scores contain no finite value")
    median = float(np.median(finite_train))
    train_imputed = np.where(np.isfinite(train), train, median)
    test_imputed = np.where(np.isfinite(test), test, median)

    order = np.argsort(train_imputed, kind="mergesort")
    sorted_train = train_imputed[order]
    ranks = np.empty(len(train_imputed), dtype=np.float64)
    start = 0
    while start < len(sorted_train):
        stop = start + 1
        while stop < len(sorted_train) and sorted_train[stop] == sorted_train[start]:
            stop += 1
        average_one_based_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_one_based_rank
        start = stop
    train_midrank = (ranks - 0.5) / len(train_imputed)

    below = np.searchsorted(sorted_train, test_imputed, side="left")
    at_or_below = np.searchsorted(sorted_train, test_imputed, side="right")
    test_midrank = (below + 0.5 * (at_or_below - below)) / len(train_imputed)
    mean = float(train_midrank.mean())
    scale = float(train_midrank.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("outer-training empirical midranks have zero variance")
    return FoldTransform(
        train=(train_midrank - mean) / scale,
        test=(test_midrank - mean) / scale,
        imputation_median=median,
        midrank_mean=mean,
        midrank_scale=scale,
    )


def _outer_training_nuisance(
    train_values: np.ndarray, test_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_values, dtype=np.float64)
    test = np.asarray(test_values, dtype=np.float64)
    if train.ndim == 1:
        train = train[:, None]
        test = test[:, None]
    train_result = np.empty_like(train)
    test_result = np.empty_like(test)
    for column in range(train.shape[1]):
        finite = train[np.isfinite(train[:, column]), column]
        if len(finite) == 0:
            raise ValueError("outer training nuisance contains no finite value")
        median = float(np.median(finite))
        train_column = np.where(np.isfinite(train[:, column]), train[:, column], median)
        test_column = np.where(np.isfinite(test[:, column]), test[:, column], median)
        mean = float(train_column.mean())
        scale = float(train_column.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("outer-training nuisance has zero variance")
        train_result[:, column] = (train_column - mean) / scale
        test_result[:, column] = (test_column - mean) / scale
    return train_result, test_result


def group_adjusted_oof(
    outcome: Sequence[float] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    experimental_group: Sequence[float] | np.ndarray,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    nuisance: np.ndarray | None = None,
) -> OOFResult:
    """Fit repeated participant-out group-adjusted OLS models.

    Every fold fits missing-score imputation, the empirical-midrank map,
    predictor standardization, nuisance transforms, and OLS coefficients using
    its outer-training participants only. Repeated test predictions are averaged
    per participant before metrics are computed.
    """

    y = np.asarray(outcome, dtype=np.float64)
    x = np.asarray(score, dtype=np.float64)
    group = np.asarray(experimental_group, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1 or group.ndim != 1:
        raise ValueError("outcome, score, and experimental_group must be vectors")
    if not (len(y) == len(x) == len(group)):
        raise ValueError("outcome, score, and experimental_group must align")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(group)):
        raise ValueError("outcome and experimental_group must be finite")
    nuisance_array = None if nuisance is None else np.asarray(nuisance, dtype=np.float64)
    if nuisance_array is not None and len(nuisance_array) != len(y):
        raise ValueError("nuisance rows must align with participants")

    sums_full = np.zeros(len(y), dtype=np.float64)
    sums_reduced = np.zeros(len(y), dtype=np.float64)
    sums_intercept = np.zeros(len(y), dtype=np.float64)
    counts = np.zeros(len(y), dtype=np.int64)
    fold_count = 0
    for train_indices, test_indices in folds:
        train = np.asarray(train_indices, dtype=np.int64)
        test = np.asarray(test_indices, dtype=np.int64)
        if train.ndim != 1 or test.ndim != 1 or len(train) == 0 or len(test) == 0:
            raise ValueError("each fold must contain nonempty train and test vectors")
        invalid_index = (
            np.any(train < 0)
            or np.any(test < 0)
            or np.any(train >= len(y))
            or np.any(test >= len(y))
        )
        if invalid_index:
            raise IndexError("fold index is outside the participant array")
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise ValueError("outer train and test indices must be unique")
        if np.intersect1d(train, test).size:
            raise ValueError("outer train and test participants overlap")

        transformed = outer_training_midrank(x[train], x[test])
        reduced_train_columns = [np.ones(len(train)), group[train]]
        reduced_test_columns = [np.ones(len(test)), group[test]]
        if nuisance_array is not None:
            nuisance_train, nuisance_test = _outer_training_nuisance(
                nuisance_array[train], nuisance_array[test]
            )
            reduced_train_columns.extend(
                nuisance_train[:, column]
                for column in range(nuisance_train.shape[1])
            )
            reduced_test_columns.extend(
                nuisance_test[:, column]
                for column in range(nuisance_test.shape[1])
            )
        reduced_train = np.column_stack(reduced_train_columns)
        reduced_test = np.column_stack(reduced_test_columns)
        full_train = np.column_stack([reduced_train, transformed.train])
        full_test = np.column_stack([reduced_test, transformed.test])
        if np.linalg.matrix_rank(reduced_train) != reduced_train.shape[1]:
            raise ValueError("outer-training reduced design is rank deficient")
        if np.linalg.matrix_rank(full_train) != full_train.shape[1]:
            raise ValueError("outer-training full design is rank deficient")

        beta_full = np.linalg.lstsq(full_train, y[train], rcond=None)[0]
        beta_reduced = np.linalg.lstsq(reduced_train, y[train], rcond=None)[0]
        intercept = float(y[train].mean())
        sums_full[test] += full_test @ beta_full
        sums_reduced[test] += reduced_test @ beta_reduced
        sums_intercept[test] += intercept
        counts[test] += 1
        fold_count += 1
    if fold_count == 0 or np.any(counts == 0):
        raise ValueError("folds must supply at least one test prediction per participant")

    prediction_full = sums_full / counts
    prediction_reduced = sums_reduced / counts
    prediction_intercept = sums_intercept / counts
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        raise ValueError("outcome has zero variance")
    sse_full = float(np.sum((y - prediction_full) ** 2))
    sse_reduced = float(np.sum((y - prediction_reduced) ** 2))
    sse_intercept = float(np.sum((y - prediction_intercept) ** 2))
    metrics = {
        "sst": sst,
        "sse_full": sse_full,
        "sse_group_or_reduced": sse_reduced,
        "sse_intercept_oof": sse_intercept,
        "full_cv_r2": 1.0 - sse_full / sst,
        "group_or_reduced_cv_r2": 1.0 - sse_reduced / sst,
        "delta_cv_r2": (sse_reduced - sse_full) / sst,
        "mae": float(np.mean(np.abs(y - prediction_full))),
        "rmse": float(np.sqrt(np.mean((y - prediction_full) ** 2))),
    }
    return OOFResult(
        prediction_full=prediction_full,
        prediction_reduced=prediction_reduced,
        prediction_intercept=prediction_intercept,
        test_counts=counts,
        metrics=metrics,
    )


__all__ = [
    "TASK_VALID_CLASSES",
    "FeedbackAggregation",
    "FoldTransform",
    "OOFResult",
    "aggregate_trial_and_seed_equal",
    "group_adjusted_oof",
    "outer_training_midrank",
    "task_valid_membership",
]
