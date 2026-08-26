from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SelfSupervisedAugmentationConfig:
    max_shift_samples: int = 16
    mask_fraction: float = 0.05
    noise_std: float = 0.03
    frequency_dropout_probability: float = 0.08
    frequency_amplitude_jitter_std: float = 0.05
    temperature: float = 0.20


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def temporal_view(
    x: torch.Tensor,
    *,
    max_shift_samples: int = 16,
    mask_fraction: float = 0.05,
    noise_std: float = 0.03,
) -> torch.Tensor:
    view = x.clone()
    if max_shift_samples > 0:
        shifts = torch.randint(
            -max_shift_samples,
            max_shift_samples + 1,
            (view.shape[0],),
            device=view.device,
        )
        view = torch.stack(
            [torch.roll(item, shifts=int(shift.item()), dims=-1) for item, shift in zip(view, shifts)],
            dim=0,
        )
    if mask_fraction > 0:
        mask_len = max(1, int(round(view.shape[-1] * mask_fraction)))
        starts = torch.randint(
            0,
            max(view.shape[-1] - mask_len + 1, 1),
            (view.shape[0],),
            device=view.device,
        )
        for index, start in enumerate(starts):
            start_index = int(start.item())
            view[index, :, start_index : start_index + mask_len] = 0.0
    if noise_std > 0:
        view = view + torch.randn_like(view) * noise_std
    return normalize_batch(view)


def frequency_view(
    x: torch.Tensor,
    *,
    dropout_probability: float = 0.08,
    amplitude_jitter_std: float = 0.05,
) -> torch.Tensor:
    spectrum = torch.fft.rfft(x, dim=-1)
    if dropout_probability > 0:
        keep = (torch.rand(spectrum.real.shape, device=x.device) > dropout_probability).to(spectrum.dtype)
        keep[..., 0] = 1.0
        spectrum = spectrum * keep
    if amplitude_jitter_std > 0:
        scale = 1.0 + torch.randn(
            spectrum.real.shape,
            device=x.device,
            dtype=x.dtype,
        ) * amplitude_jitter_std
        spectrum = spectrum * scale.clamp(0.5, 1.5).to(spectrum.dtype)
    return normalize_batch(torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1))


def symmetric_contrastive_loss(
    time_embeddings: torch.Tensor,
    frequency_embeddings: torch.Tensor,
    temperature: float = 0.20,
) -> torch.Tensor:
    time_embeddings = nn.functional.normalize(time_embeddings, dim=1)
    frequency_embeddings = nn.functional.normalize(frequency_embeddings, dim=1)
    logits = time_embeddings @ frequency_embeddings.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels)
        + nn.functional.cross_entropy(logits.T, labels)
    )


def binary_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    labels = labels.reshape(-1).to(torch.long)
    if int(torch.unique(labels).numel()) < 2:
        return embeddings.new_tensor(0.0)
    embeddings = nn.functional.normalize(embeddings, dim=1)
    positive_mask = labels[:, None].eq(labels[None, :]).float()
    self_mask = torch.eye(len(labels), dtype=torch.float32, device=embeddings.device)
    positive_mask = positive_mask * (1.0 - self_mask)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return embeddings.new_tensor(0.0)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob = (positive_mask[valid] * log_prob[valid]).sum(dim=1) / positive_count[valid]
    return -mean_log_prob.mean()


def structure_supervised_contrastive_loss(
    record_embeddings: torch.Tensor,
    binary_structure_labels: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    losses = [
        binary_supervised_contrastive_loss(
            record_embeddings,
            binary_structure_labels[:, column],
            temperature,
        )
        for column in range(binary_structure_labels.shape[1])
    ]
    return torch.stack(losses).mean() if losses else record_embeddings.new_tensor(0.0)


def paired_bag_consistency_loss(record_embeddings: torch.Tensor) -> torch.Tensor:
    if record_embeddings.shape[0] < 2 or record_embeddings.shape[0] % 2:
        return record_embeddings.new_tensor(0.0)
    first = record_embeddings[0::2]
    second = record_embeddings[1::2]
    return (1.0 - nn.functional.cosine_similarity(first, second, dim=1)).mean()


def sleep_structure_objective(
    stage_logits: torch.Tensor,
    stage_targets: torch.Tensor,
    record_embeddings: torch.Tensor,
    binary_structure_labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    structure_weight: float = 0.05,
    consistency_weight: float = 0.03,
    temperature: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stage_loss = nn.functional.cross_entropy(stage_logits, stage_targets, weight=class_weights)
    structure_loss = structure_supervised_contrastive_loss(
        record_embeddings,
        binary_structure_labels,
        temperature,
    )
    consistency_loss = paired_bag_consistency_loss(record_embeddings)
    total = stage_loss + structure_weight * structure_loss + consistency_weight * consistency_loss
    return total, {
        "stage": stage_loss,
        "structure": structure_loss,
        "bag_consistency": consistency_loss,
    }

