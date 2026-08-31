from __future__ import annotations

import json

import numpy as np

from sleep2mi.longitudinal import (
    aggregate_trial_and_seed_equal,
    group_adjusted_oof,
    task_valid_membership,
)


def main() -> int:
    """Exercise the public longitudinal evaluator without participant data."""

    rng = np.random.default_rng(67)
    participants = np.arange(18)
    group = np.tile([0.0, 1.0], len(participants) // 2)
    latent = np.linspace(0.15, 0.85, len(participants))

    rows: list[list[float]] = []
    targets: list[int] = []
    participant_ids: list[int] = []
    tasks: list[str] = []
    seeds: list[str] = []
    trials: list[int] = []
    for participant, value in zip(participants, latent, strict=True):
        for seed_index, seed in enumerate(("seed_a", "seed_b")):
            for trial in range(4):
                target = 1 if trial % 2 == 0 else 2
                noisy_value = float(np.clip(value + rng.normal(0.0, 0.02), 0.02, 0.98))
                valid = [0.8 * noisy_value, 0.8 * (1.0 - noisy_value)]
                if target == 2:
                    valid.reverse()
                rows.append([valid[0], valid[1], 0.10, 0.10])
                targets.append(target)
                participant_ids.append(int(participant))
                tasks.append("LR")
                seeds.append(seed)
                trials.append(seed_index * 10 + trial)

    scores = task_valid_membership(np.asarray(rows), targets, tasks)
    aggregation = aggregate_trial_and_seed_equal(
        scores,
        participant_ids,
        tasks,
        seeds,
        trials,
    )
    participant_score = np.asarray(
        [aggregation.ensemble_scores[(int(participant), "LR")] for participant in participants]
    )
    outcome = 0.25 * group + 1.20 * latent + rng.normal(0.0, 0.04, len(participants))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in range(3):
        test = participants[offset::3]
        train = np.setdiff1d(participants, test)
        folds.append((train, test))
    result = group_adjusted_oof(outcome, participant_score, group, folds)

    summary = {
        "data": "synthetic",
        "participants": len(participants),
        "task": "LR",
        "seeds": 2,
        "trials_per_participant_seed": 4,
        "minimum_test_predictions_per_participant": int(result.test_counts.min()),
        "metrics": {key: float(value) for key, value in result.metrics.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
