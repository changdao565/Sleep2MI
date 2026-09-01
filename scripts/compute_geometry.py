from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sleep2mi.geometry import (
    geometry_ratio,
    pca2_space,
    permutation_test,
    stratified_bootstrap_interval,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Sleep2MI participant PCA2 geometry statistic, "
            "permutation p value, and stratified bootstrap interval."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="participant-by-feature .npy or .npz array",
    )
    parser.add_argument(
        "--feature-key",
        default="embeddings",
        help="array key when --features is .npz (default: embeddings)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="binary fixed-group labels in .npy or .npz format",
    )
    parser.add_argument(
        "--label-key",
        default="labels",
        help="array key when --labels is .npz (default: labels)",
    )
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--coordinates-output", type=Path, default=None)
    return parser.parse_args(argv)


def _load_array(path: Path, key: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key not in loaded.files:
                available = ", ".join(sorted(loaded.files))
                raise KeyError(
                    f"{path} has no array {key!r}; available keys: {available}"
                )
            array = np.asarray(loaded[key])
        finally:
            loaded.close()
        return array
    return np.asarray(loaded)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    if args.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    features = _load_array(args.features, args.feature_key)
    labels = _load_array(args.labels, args.label_key).reshape(-1)
    if features.ndim != 2:
        raise ValueError("features must have shape (participants, features)")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must contain the same participants")
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("features must be numeric")
    if np.isinf(features).any():
        raise ValueError("features contain infinite values")
    if not np.isfinite(labels).all():
        raise ValueError("labels contain non-finite values")
    unique_labels = set(np.unique(labels).tolist())
    if unique_labels != {0, 1}:
        raise ValueError("labels must contain both binary groups 0 and 1")

    coordinates, explained = pca2_space(features)
    observed = geometry_ratio(coordinates, labels)
    p_value, _ = permutation_test(
        coordinates,
        labels,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    lower, upper = stratified_bootstrap_interval(
        coordinates,
        labels,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    result: dict[str, object] = {
        "bootstrap_95_ci": [lower, upper],
        "geometry_ratio": observed,
        "n_bootstrap": int(args.n_bootstrap),
        "n_features": int(features.shape[1]),
        "n_participants": int(features.shape[0]),
        "n_permutations": int(args.n_permutations),
        "pca2_explained_variance": explained,
        "permutation_p_value": p_value,
        "seed": int(args.seed),
    }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8", newline="\n")
    if args.coordinates_output is not None:
        args.coordinates_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.coordinates_output,
            coordinates=coordinates,
            labels=labels,
        )
    print(rendered, end="")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
