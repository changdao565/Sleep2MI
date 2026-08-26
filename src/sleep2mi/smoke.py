from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from .geometry import geometry_ratio, pca2_space
from .model import Sleep2MIConfig, build_sleep2mi
from .objectives import (
    frequency_view,
    sleep_structure_objective,
    symmetric_contrastive_loss,
    temporal_view,
)
from .pooling import BagAttentionPooling


def run_synthetic_smoke(seed: int = 67) -> dict[str, float | int]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = Sleep2MIConfig()
    model = build_sleep2mi(config).eval()

    signals = torch.randn(8, 1, 3000)
    with torch.no_grad():
        logits, embeddings = model(signals)
        time_embeddings = model.encode(temporal_view(signals))
        frequency_embeddings = model.encode(frequency_view(signals))
    assert logits.shape == (8, 5)
    assert embeddings.shape == (8, 32)

    contrastive = symmetric_contrastive_loss(time_embeddings, frequency_embeddings)
    bag_tokens = embeddings.reshape(4, 2, 32)
    record_embeddings = BagAttentionPooling()(bag_tokens)
    paired_records = torch.repeat_interleave(record_embeddings, repeats=2, dim=0)
    structure_labels = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    stage_targets = torch.arange(8) % 5
    objective, _ = sleep_structure_objective(
        logits,
        stage_targets,
        paired_records,
        structure_labels,
    )

    features = rng.normal(size=(20, 32))
    labels = np.repeat([0, 1], 10)
    features[labels == 1, :3] += 0.5
    coordinates, explained = pca2_space(features)
    geometry = geometry_ratio(coordinates, labels)

    encoder_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("stage_head")
    )
    assert encoder_parameters == 68322
    for value in (contrastive.item(), objective.item(), explained, geometry):
        assert np.isfinite(value)

    return {
        "encoder_parameters": encoder_parameters,
        "embedding_dimension": int(embeddings.shape[1]),
        "contrastive_loss": float(contrastive.item()),
        "sleep_structure_objective": float(objective.item()),
        "pca2_explained_variance": explained,
        "geometry_ratio": geometry,
    }


def main() -> None:
    print(json.dumps(run_synthetic_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
