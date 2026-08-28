from __future__ import annotations

import json
from pathlib import Path

import torch

from sleep2mi import Sleep2MIConfig, build_sleep2mi


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "sleep2mi.json"


def main() -> None:
    torch.manual_seed(0)
    configuration_values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = Sleep2MIConfig.from_json(CONFIG_PATH)
    model = build_sleep2mi(config).eval()

    sampling_rate_hz = int(configuration_values["sampling_rate_hz"])
    epoch_seconds = int(configuration_values["epoch_seconds"])
    synthetic_sleep_eeg = torch.randn(
        2,
        1,
        sampling_rate_hz * epoch_seconds,
    )

    with torch.no_grad():
        stage_logits, embeddings = model(synthetic_sleep_eeg)

    encoder_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("stage_head")
    )
    summary = {
        "configuration": CONFIG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "input_shape": list(synthetic_sleep_eeg.shape),
        "stage_logits_shape": list(stage_logits.shape),
        "embedding_shape": list(embeddings.shape),
        "encoder_parameters": encoder_parameters,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
