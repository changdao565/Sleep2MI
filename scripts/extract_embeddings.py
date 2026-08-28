from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from sleep2mi import Sleep2MIConfig, build_sleep2mi


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "sleep2mi.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Sleep2MI stage logits and 32-dimensional embeddings from "
            "a NumPy EEG-epoch array using a trusted checkpoint."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=".npy array with shape (epochs, channels, samples)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted PyTorch checkpoint containing model_state",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output .npz file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "configuration JSON; by default use checkpoint['config'] when "
            "available, otherwise configs/sleep2mi.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    return parser.parse_args(argv)


def _load_config(
    payload: Mapping[str, object],
    config_path: Path | None,
) -> Sleep2MIConfig:
    if config_path is not None:
        return Sleep2MIConfig.from_json(config_path)
    checkpoint_config = payload.get("config")
    if isinstance(checkpoint_config, Mapping):
        return Sleep2MIConfig.from_dict(checkpoint_config)
    return Sleep2MIConfig.from_json(DEFAULT_CONFIG_PATH)


def _load_model(
    checkpoint_path: Path,
    config_path: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise KeyError("checkpoint must contain a 'model_state' mapping")
    model = build_sleep2mi(_load_config(payload, config_path))
    model.load_state_dict(model_state, strict=True)
    return model.to(device).eval()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    signals = np.load(args.input, allow_pickle=False, mmap_mode="r")
    if not isinstance(signals, np.ndarray) or signals.ndim != 3:
        raise ValueError("input must have shape (epochs, channels, samples)")
    if signals.shape[0] < 1 or signals.shape[1] < 1 or signals.shape[2] < 1:
        raise ValueError("input dimensions must be nonzero")
    if not np.issubdtype(signals.dtype, np.number):
        raise TypeError("input array must be numeric")

    device = _resolve_device(args.device)
    model = _load_model(args.checkpoint, args.config, device)
    embedding_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, signals.shape[0], args.batch_size):
            stop = min(start + args.batch_size, signals.shape[0])
            batch_array = np.array(
                signals[start:stop],
                dtype=np.float32,
                copy=True,
            )
            if not np.isfinite(batch_array).all():
                raise ValueError("input contains non-finite values")
            batch = torch.from_numpy(batch_array).to(device)
            logits, embeddings = model(batch)
            logit_batches.append(logits.cpu().numpy())
            embedding_batches.append(embeddings.cpu().numpy())

    stage_logits = np.concatenate(logit_batches, axis=0)
    embeddings = np.concatenate(embedding_batches, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        stage_logits=stage_logits,
    )
    return {
        "device": str(device),
        "embedding_dimension": int(embeddings.shape[1]),
        "input_shape": list(signals.shape),
        "n_epochs": int(signals.shape[0]),
        "output": str(args.output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
