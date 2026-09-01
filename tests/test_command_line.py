from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from sleep2mi import Sleep2MIConfig, build_sleep2mi


REPOSITORY_ROOT = Path(__file__).parents[1]


def run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_extract_embeddings_cli_strictly_loads_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(67)
    config = Sleep2MIConfig()
    model = build_sleep2mi(config).eval()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )
    input_path = tmp_path / "epochs.npy"
    rng = np.random.default_rng(67)
    np.save(input_path, rng.normal(size=(3, 1, 400)).astype(np.float32))
    output_path = tmp_path / "embeddings.npz"

    completed = run_script(
        "scripts/extract_embeddings.py",
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(output_path),
        "--batch-size",
        "2",
        "--device",
        "cpu",
    )

    summary = json.loads(completed.stdout)
    assert summary["embedding_dimension"] == 32
    assert summary["input_shape"] == [3, 1, 400]
    with np.load(output_path, allow_pickle=False) as output:
        assert output["embeddings"].shape == (3, 32)
        assert output["stage_logits"].shape == (3, 5)
        assert np.isfinite(output["embeddings"]).all()
        assert np.isfinite(output["stage_logits"]).all()


def test_compute_geometry_cli_writes_machine_readable_outputs(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(67)
    labels = np.repeat([0, 1], 10)
    features = rng.normal(size=(20, 8))
    features[labels == 1, :2] += 0.5
    features[0, 0] = np.nan
    features_path = tmp_path / "participant_embeddings.npz"
    labels_path = tmp_path / "fixed_groups.npy"
    np.savez_compressed(features_path, embeddings=features)
    np.save(labels_path, labels)
    output_json = tmp_path / "geometry.json"
    coordinates_output = tmp_path / "coordinates.npz"

    completed = run_script(
        "scripts/compute_geometry.py",
        "--features",
        str(features_path),
        "--labels",
        str(labels_path),
        "--n-permutations",
        "99",
        "--n-bootstrap",
        "99",
        "--seed",
        "67",
        "--output-json",
        str(output_json),
        "--coordinates-output",
        str(coordinates_output),
    )

    stdout_result = json.loads(completed.stdout)
    file_result = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_result == file_result
    assert file_result["n_participants"] == 20
    assert file_result["n_features"] == 8
    assert 0.0 <= file_result["permutation_p_value"] <= 1.0
    assert len(file_result["bootstrap_95_ci"]) == 2
    with np.load(coordinates_output, allow_pickle=False) as output:
        assert output["coordinates"].shape == (20, 2)
        assert np.isfinite(output["coordinates"]).all()
        assert np.array_equal(output["labels"], labels)


def test_longitudinal_synthetic_cli_reports_finite_metrics() -> None:
    completed = run_script("scripts/run_longitudinal_synthetic.py")
    result = json.loads(completed.stdout)
    assert result["data"] == "synthetic"
    assert result["participants"] == 18
    assert result["task"] == "LR"
    assert result["seeds"] == 2
    assert result["minimum_test_predictions_per_participant"] == 1
    assert np.isfinite(list(result["metrics"].values())).all()
