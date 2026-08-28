import json
from pathlib import Path

import pytest
import torch

from sleep2mi import Sleep2MIConfig, Sleep2MIEncoder


LEGACY_V1_CONFIG = {
    "n_classes": 5,
    "d_model": 96,
    "cnn_channels": 32,
    "kernel_sizes": [15, 31, 63],
    "temporal_stride": 8,
    "transformer_layers": 1,
    "transformer_heads": 4,
    "dropout": 0.15,
    "use_transformer": True,
    "channel_pool": "attention",
    "sequence_model": "gru",
    "bottleneck_dim": 32,
}


def test_legacy_v1_config_normalizes_from_dict_and_json(tmp_path: Path) -> None:
    source = dict(LEGACY_V1_CONFIG)
    from_dict = Sleep2MIConfig.from_dict(source)
    assert source == LEGACY_V1_CONFIG
    assert from_dict.to_dict() == {
        "n_classes": 5,
        "d_model": 96,
        "cnn_channels": 32,
        "kernel_sizes": (15, 31, 63),
        "temporal_stride": 8,
        "sequence_layers": 1,
        "dropout": 0.15,
        "channel_pool": "attention",
        "sequence_model": "gru",
        "bottleneck_dim": 32,
    }

    config_path = tmp_path / "legacy_v1_config.json"
    config_path.write_text(json.dumps(LEGACY_V1_CONFIG), encoding="utf-8")
    assert Sleep2MIConfig.from_json(config_path) == from_dict


def test_legacy_v1_checkpoint_strict_load_is_bit_exact(tmp_path: Path) -> None:
    torch.manual_seed(67)
    reference = Sleep2MIEncoder(Sleep2MIConfig.from_dict(LEGACY_V1_CONFIG)).eval()
    checkpoint_path = tmp_path / "legacy_v1_checkpoint.pt"
    torch.save(
        {
            "config": dict(LEGACY_V1_CONFIG),
            "model_state": reference.state_dict(),
        },
        checkpoint_path,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = Sleep2MIEncoder(Sleep2MIConfig.from_dict(payload["config"])).eval()
    incompatible = reloaded.load_state_dict(payload["model_state"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert list(reloaded.state_dict()) == list(payload["model_state"])
    assert len(payload["model_state"]) == 53
    for name, expected in payload["model_state"].items():
        assert torch.equal(reloaded.state_dict()[name], expected), name

    generator = torch.Generator().manual_seed(6701)
    inputs = torch.randn(2, 3, 400, generator=generator)
    with torch.inference_mode():
        expected_logits, expected_embedding = reference(inputs)
        actual_logits, actual_embedding = reloaded(inputs)
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_embedding, expected_embedding)


def test_legacy_and_canonical_layer_names_must_agree() -> None:
    with pytest.raises(ValueError, match="Conflicting sequence_layers"):
        Sleep2MIConfig.from_dict(
            {
                **LEGACY_V1_CONFIG,
                "sequence_layers": 2,
            }
        )


@pytest.mark.parametrize("field", ["transformer_width", "checkpoint_path"])
def test_unknown_checkpoint_config_fields_remain_errors(field: str) -> None:
    with pytest.raises(ValueError, match="Unknown Sleep2MI configuration field"):
        Sleep2MIConfig.from_dict({**LEGACY_V1_CONFIG, field: "unexpected"})
