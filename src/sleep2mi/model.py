from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn


_NON_MODEL_CONFIG_FIELDS = frozenset(
    {
        "sampling_rate_hz",
        "epoch_seconds",
        "structure_loss_weight",
        "bag_consistency_weight",
        "structure_temperature",
        "self_supervised_temperature",
        "temporal_max_shift_samples",
        "temporal_mask_fraction",
        "temporal_noise_std",
        "frequency_dropout_probability",
        "frequency_amplitude_jitter_std",
        "self_supervised_training",
    }
)

_LEGACY_MODEL_CONFIG_FIELDS = frozenset(
    {
        "transformer_layers",
        "transformer_heads",
        "use_transformer",
    }
)


@dataclass(frozen=True)
class Sleep2MIConfig:
    n_classes: int = 5
    d_model: int = 96
    cnn_channels: int = 32
    kernel_sizes: tuple[int, ...] = (15, 31, 63)
    temporal_stride: int = 8
    sequence_layers: int = 1
    dropout: float = 0.15
    channel_pool: str = "attention"
    sequence_model: str = "gru"
    bottleneck_dim: int = 32

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "Sleep2MIConfig":
        """Extract model fields and normalize the preserved V1 vocabulary.

        The released V1 checkpoints predate the canonical ``sequence_layers``
        name.  Their explicit GRU configuration stores the same value under
        ``transformer_layers`` and also carries two transformer-only fields
        that are inert for that GRU runtime.  These three known legacy fields
        are resolved here; any other unregistered field remains an error.
        """
        normalized = dict(values)
        model_fields = {field.name for field in fields(cls)}
        unknown_fields = (
            set(normalized)
            - model_fields
            - _NON_MODEL_CONFIG_FIELDS
            - _LEGACY_MODEL_CONFIG_FIELDS
        )
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown Sleep2MI configuration field(s): {names}")

        legacy_use_transformer = normalized.pop("use_transformer", None)
        if legacy_use_transformer is not None and not isinstance(
            legacy_use_transformer, bool
        ):
            raise TypeError("legacy use_transformer must be a boolean")

        legacy_heads = normalized.pop("transformer_heads", None)
        if legacy_heads is not None:
            if isinstance(legacy_heads, bool) or not isinstance(legacy_heads, int):
                raise TypeError("legacy transformer_heads must be an integer")
            if legacy_heads <= 0:
                raise ValueError("legacy transformer_heads must be positive")

        legacy_layers = normalized.pop("transformer_layers", None)
        canonical_layers = normalized.get("sequence_layers")
        if legacy_layers is not None:
            if isinstance(legacy_layers, bool) or not isinstance(legacy_layers, int):
                raise TypeError("legacy transformer_layers must be an integer")
            if legacy_layers <= 0:
                raise ValueError("legacy transformer_layers must be positive")
        if canonical_layers is not None and legacy_layers is not None:
            if canonical_layers != legacy_layers:
                raise ValueError(
                    "Conflicting sequence_layers and legacy transformer_layers"
                )
        elif canonical_layers is None and legacy_layers is not None:
            normalized["sequence_layers"] = legacy_layers

        raw_sequence_model = normalized.get("sequence_model")
        if raw_sequence_model == "auto":
            use_transformer = (
                True
                if legacy_use_transformer is None
                else legacy_use_transformer
            )
            normalized["sequence_model"] = (
                "transformer" if use_transformer else "none"
            )

        model_values = {
            key: value for key, value in normalized.items() if key in model_fields
        }
        if "kernel_sizes" in model_values:
            kernel_sizes = model_values["kernel_sizes"]
            if not isinstance(kernel_sizes, (list, tuple)):
                raise TypeError(
                    "kernel_sizes must be a list or tuple of positive integers"
                )
            if not kernel_sizes:
                raise ValueError(
                    "kernel_sizes must contain at least one positive integer"
                )
            for index, kernel_size in enumerate(kernel_sizes):
                if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
                    raise TypeError(
                        f"kernel_sizes[{index}] must be an integer; "
                        f"got {type(kernel_size).__name__}"
                    )
                if kernel_size <= 0:
                    raise ValueError(
                        f"kernel_sizes[{index}] must be positive; got {kernel_size}"
                    )
            model_values["kernel_sizes"] = tuple(kernel_sizes)
        return cls(**model_values)

    @classmethod
    def from_json(cls, path: str | Path) -> "Sleep2MIConfig":
        """Load model fields from a JSON manuscript configuration file."""
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise TypeError("Sleep2MI configuration JSON must contain an object")
        return cls.from_dict(values)


class MultiScaleTemporalCNN(nn.Module):
    def __init__(self, config: Sleep2MIConfig):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        1,
                        config.cnn_channels,
                        kernel_size=kernel_size,
                        stride=config.temporal_stride,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(config.cnn_channels),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                for kernel_size in config.kernel_sizes
            ]
        )
        self.proj = nn.Sequential(
            nn.Conv1d(config.cnn_channels * len(config.kernel_sizes), config.d_model, 1),
            nn.BatchNorm1d(config.d_model),
            nn.GELU(),
        )

    def forward_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = x.shape
        x = x.reshape(batch * channels, 1, samples)
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        features = self.proj(features)
        return features.reshape(batch, channels, features.shape[1], features.shape[2])


class ChannelAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_summary = features.mean(dim=-1)
        weights = torch.softmax(self.score(channel_summary), dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens), dim=1)
        return (tokens * weights).sum(dim=1)


class Sleep2MIEncoder(nn.Module):
    def __init__(self, config: Sleep2MIConfig):
        super().__init__()
        self.config = config
        self.cnn = MultiScaleTemporalCNN(config)
        if config.channel_pool == "mean":
            self.channel_pool: nn.Module | None = None
        elif config.channel_pool == "attention":
            self.channel_pool = ChannelAttentionPooling(config.d_model)
        else:
            raise ValueError(f"Unsupported channel_pool: {config.channel_pool}")

        if config.sequence_model == "gru":
            self.gru: nn.GRU | None = nn.GRU(
                config.d_model,
                config.d_model // 2,
                num_layers=config.sequence_layers,
                dropout=config.dropout if config.sequence_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
        elif config.sequence_model == "none":
            self.gru = None
        else:
            raise ValueError(f"Unsupported sequence_model: {config.sequence_model}")

        self.pool = AttentionPooling(config.d_model)
        if config.bottleneck_dim != config.d_model:
            self.embedding_proj: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.bottleneck_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.embedding_proj = None
        self.stage_head = nn.Sequential(
            nn.LayerNorm(config.bottleneck_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.bottleneck_dim, config.n_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        per_channel = self.cnn.forward_per_channel(x)
        pooled_channels = (
            per_channel.mean(dim=1)
            if self.channel_pool is None
            else self.channel_pool(per_channel)
        )
        tokens = pooled_channels.transpose(1, 2)
        if self.gru is not None:
            tokens, _ = self.gru(tokens)
        embedding = self.pool(tokens)
        if self.embedding_proj is not None:
            embedding = self.embedding_proj(embedding)
        return embedding

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x)
        return self.stage_head(embedding), embedding


def build_sleep2mi(config: Sleep2MIConfig | None = None) -> Sleep2MIEncoder:
    return Sleep2MIEncoder(config or Sleep2MIConfig())
