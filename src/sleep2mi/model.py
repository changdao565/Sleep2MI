from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


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
    def from_dict(cls, values: dict[str, object]) -> "Sleep2MIConfig":
        values = dict(values)
        if "kernel_sizes" in values and not isinstance(values["kernel_sizes"], tuple):
            values["kernel_sizes"] = tuple(values["kernel_sizes"])  # type: ignore[arg-type]
        return cls(**values)


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

