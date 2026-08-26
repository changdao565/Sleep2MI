from __future__ import annotations

import torch
from torch import nn


class BagAttentionPooling(nn.Module):
    """Attention pooling over local EEG-epoch embeddings."""

    def __init__(self, embedding_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        hidden = max(embedding_dim // 2, 8)
        self.score = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(tokens).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)

