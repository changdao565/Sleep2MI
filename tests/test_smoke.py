from pathlib import Path

import pytest

from sleep2mi import Sleep2MIConfig
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
