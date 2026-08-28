import inspect
import json
from pathlib import Path

import pytest

from sleep2mi import Sleep2MIConfig
from sleep2mi.objectives import (
    SelfSupervisedAugmentationConfig,
    sleep_structure_objective,
    symmetric_contrastive_loss,
)
from sleep2mi.smoke import run_synthetic_smoke


MANUSCRIPT_CONFIG = Path(__file__).parents[1] / "configs" / "sleep2mi.json"


def test_synthetic_smoke() -> None:
    result = run_synthetic_smoke(seed=67)
    assert result["encoder_parameters"] == 68322
    assert result["embedding_dimension"] == 32


def test_manuscript_config_loads_through_public_interface() -> None:
    config = Sleep2MIConfig.from_json(MANUSCRIPT_CONFIG)
    assert config.to_dict() == {
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


def test_objective_defaults_match_manuscript_config() -> None:
    values = json.loads(MANUSCRIPT_CONFIG.read_text(encoding="utf-8"))
    augmentations = SelfSupervisedAugmentationConfig()
    assert augmentations.max_shift_samples == values["temporal_max_shift_samples"]
    assert augmentations.mask_fraction == values["temporal_mask_fraction"]
    assert augmentations.noise_std == values["temporal_noise_std"]
    assert (
        augmentations.frequency_dropout_probability
        == values["frequency_dropout_probability"]
    )
    assert (
        augmentations.frequency_amplitude_jitter_std
        == values["frequency_amplitude_jitter_std"]
    )
    assert augmentations.temperature == values["self_supervised_temperature"]

    structure_signature = inspect.signature(sleep_structure_objective)
    assert (
        structure_signature.parameters["structure_weight"].default
        == values["structure_loss_weight"]
    )
    assert (
        structure_signature.parameters["consistency_weight"].default
        == values["bag_consistency_weight"]
    )
    assert (
        structure_signature.parameters["temperature"].default
        == values["structure_temperature"]
    )
    self_supervised_signature = inspect.signature(symmetric_contrastive_loss)
    assert (
        self_supervised_signature.parameters["temperature"].default
        == values["self_supervised_temperature"]
    )


def test_from_dict_converts_kernel_sizes_to_tuple() -> None:
    config = Sleep2MIConfig.from_dict({"kernel_sizes": [7, 15, 31]})
    assert config.kernel_sizes == (7, 15, 31)


@pytest.mark.parametrize("field", ["d_modle", "sampling_rate_hzz"])
def test_from_dict_rejects_unregistered_fields(field: str) -> None:
    with pytest.raises(ValueError, match="Unknown Sleep2MI configuration field"):
        Sleep2MIConfig.from_dict({field: 96})


@pytest.mark.parametrize(
    ("kernel_size", "error_type", "message"),
    [
        (True, TypeError, r"kernel_sizes\[1\] must be an integer; got bool"),
        ("31", TypeError, r"kernel_sizes\[1\] must be an integer; got str"),
        (0, ValueError, r"kernel_sizes\[1\] must be positive; got 0"),
        (-31, ValueError, r"kernel_sizes\[1\] must be positive; got -31"),
    ],
)
def test_from_dict_rejects_invalid_kernel_sizes(
    kernel_size: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        Sleep2MIConfig.from_dict({"kernel_sizes": [15, kernel_size]})
