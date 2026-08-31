from __future__ import annotations

import numpy as np
import pytest

from sleep2mi.longitudinal import (
    aggregate_trial_and_seed_equal,
    group_adjusted_oof,
    outer_training_midrank,
    task_valid_membership,
)


def test_task_valid_membership_renormalizes_within_each_task() -> None:
    memberships = np.asarray(
        [
            [0.10, 0.30, 0.20, 0.40],
            [0.10, 0.30, 0.20, 0.40],
            [0.10, 0.30, 0.20, 0.40],
            [0.10, 0.30, 0.20, 0.40],
            [0.10, 0.30, 0.20, 0.40],
        ]
    )
    observed = task_valid_membership(
        memberships,
        target_classes=[1, 2, 3, 4, 4],
        tasks=["LR", "LR", "UD", "UD", "2D"],
    )
    assert observed == pytest.approx([0.25, 0.75, 1 / 3, 2 / 3, 0.40])


def test_task_valid_membership_rejects_invalid_task_target_and_simplex() -> None:
    membership = np.asarray([[0.25, 0.25, 0.25, 0.25]])
    with pytest.raises(ValueError, match="unknown task"):
        task_valid_membership(membership, [1], ["UNKNOWN"])
    with pytest.raises(ValueError, match="invalid for task LR"):
        task_valid_membership(membership, [3], ["LR"])
    with pytest.raises(ValueError, match="sum to one"):
        task_valid_membership(membership * 2, [1], ["LR"])


def test_trial_and_seed_aggregation_uses_equal_weights() -> None:
    aggregation = aggregate_trial_and_seed_equal(
        scores=[1.0, 5.0, 9.0],
        participant_ids=["P01", "P01", "P01"],
        tasks=["LR", "LR", "LR"],
        seeds=["seed_a", "seed_a", "seed_b"],
        trial_ids=["T1", "T2", "T1"],
    )
    # The two trial-level scores in seed_a receive equal weight.
    assert aggregation.seed_scores[("P01", "LR", "seed_a")] == pytest.approx(3.0)
    assert aggregation.seed_scores[("P01", "LR", "seed_b")] == pytest.approx(9.0)
    # The two predefined seed summaries then receive equal weight.
    assert aggregation.ensemble_scores[("P01", "LR")] == pytest.approx(6.0)


def test_trial_aggregation_rejects_window_level_score_duplicates() -> None:
    with pytest.raises(ValueError, match="one task-valid score per trial"):
        aggregate_trial_and_seed_equal(
            scores=[0.25, 0.75],
            participant_ids=["P01", "P01"],
            tasks=["LR", "LR"],
            seeds=["seed_a", "seed_a"],
            trial_ids=["T1", "T1"],
        )


def test_outer_training_midrank_matches_locked_definition() -> None:
    transformed = outer_training_midrank(
        train_scores=[1.0, 1.0, 3.0, np.nan],
        test_scores=[1.0, 2.0, 4.0, np.nan],
    )
    expected_train_midrank = np.asarray([0.375, 0.375, 0.875, 0.375])
    expected_test_midrank = np.asarray([0.375, 0.75, 1.0, 0.375])
    expected_mean = float(expected_train_midrank.mean())
    expected_scale = float(expected_train_midrank.std(ddof=0))
    assert transformed.imputation_median == pytest.approx(1.0)
    assert transformed.midrank_mean == pytest.approx(expected_mean)
    assert transformed.midrank_scale == pytest.approx(expected_scale)
    assert transformed.train == pytest.approx(
        (expected_train_midrank - expected_mean) / expected_scale
    )
    assert transformed.test == pytest.approx(
        (expected_test_midrank - expected_mean) / expected_scale
    )


def test_test_values_do_not_change_outer_training_transform() -> None:
    reference = outer_training_midrank([0.1, 0.3, 0.7, 0.9], [0.2])
    perturbed = outer_training_midrank([0.1, 0.3, 0.7, 0.9], [200.0])
    assert np.array_equal(reference.train, perturbed.train)
    assert reference.imputation_median == perturbed.imputation_median
    assert reference.midrank_mean == perturbed.midrank_mean
    assert reference.midrank_scale == perturbed.midrank_scale


def test_group_adjusted_oof_averages_repeated_participant_out_predictions() -> None:
    rng = np.random.default_rng(67)
    n_participants = 18
    score = np.linspace(0.05, 0.95, n_participants)
    group = np.tile([0.0, 1.0], n_participants // 2)
    outcome = 0.30 * group + 1.40 * score + rng.normal(0.0, 0.03, n_participants)
    nuisance = rng.normal(0.0, 1.0, n_participants)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    indices = np.arange(n_participants)
    for repeat in range(2):
        shifted = np.roll(indices, repeat * 3)
        for test in np.array_split(shifted, 3):
            train = np.setdiff1d(indices, test)
            folds.append((train, test))

    result = group_adjusted_oof(
        outcome,
        score,
        group,
        folds,
        nuisance=nuisance,
    )
    assert np.array_equal(result.test_counts, np.full(n_participants, 2))
    assert np.isfinite(result.prediction_full).all()
    assert np.isfinite(result.prediction_reduced).all()
    assert result.metrics["delta_cv_r2"] == pytest.approx(
        (result.metrics["sse_group_or_reduced"] - result.metrics["sse_full"])
        / result.metrics["sst"]
    )
    assert (
        result.metrics["full_cv_r2"]
        > result.metrics["group_or_reduced_cv_r2"]
    )


def test_group_adjusted_oof_rejects_leaking_or_incomplete_folds() -> None:
    outcome = np.arange(6, dtype=float)
    score = np.linspace(0.1, 0.9, 6)
    group = np.asarray([0, 1, 0, 1, 0, 1], dtype=float)
    with pytest.raises(ValueError, match="overlap"):
        group_adjusted_oof(outcome, score, group, [([0, 1, 2], [2, 3])])
    with pytest.raises(ValueError, match="at least one test prediction"):
        group_adjusted_oof(outcome, score, group, [([0, 1, 2], [3, 4])])
