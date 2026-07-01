"""Loss assembly for identity-aware Gaussian training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gsegmenter.training.object_field import GaussianIdentityField
from gsegmenter.training.regularization import identity_spatial_consistency_loss


@dataclass(slots=True)
class IdentityLossBreakdown:
    """Named losses for a Gaussian Grouping style training step."""

    total: torch.Tensor
    cross_entropy: torch.Tensor
    spatial_consistency: torch.Tensor
    valid_pixel_count: int


def compute_balanced_class_weights(
    target_labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    balance_power: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute inverse-frequency weights from the current batch labels."""

    valid_labels = target_labels[target_labels != ignore_index]
    if valid_labels.numel() == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=target_labels.device)

    counts = torch.bincount(valid_labels.long(), minlength=num_classes).to(torch.float32)
    weights = torch.ones_like(counts)
    nonzero = counts > 0
    if nonzero.any():
        ref_count = counts[nonzero].mean()
        weights[nonzero] = (ref_count / (counts[nonzero] + eps)).pow(float(balance_power))
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean().clamp_min(eps)
    return weights


def balanced_focal_cross_entropy(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    ignore_index: int,
    balance_power: float = 0.5,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy with optional inverse-frequency balancing and focal modulation."""

    num_classes = int(logits.shape[1])
    class_weights = compute_balanced_class_weights(
        target_labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
        balance_power=balance_power,
    )
    per_pixel_ce = F.cross_entropy(
        logits,
        target_labels.long(),
        ignore_index=ignore_index,
        weight=class_weights,
        reduction="none",
    )

    valid_mask = target_labels != ignore_index
    if not valid_mask.any():
        return logits.new_zeros(())

    per_pixel_ce = per_pixel_ce[valid_mask]
    if focal_gamma > 0.0:
        valid_logits = logits.permute(0, 2, 3, 1)[valid_mask]
        valid_targets = target_labels[valid_mask].long()
        probabilities = torch.softmax(valid_logits, dim=-1)
        target_probabilities = probabilities.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)
        modulation = (1.0 - target_probabilities).clamp_min(0.0).pow(float(focal_gamma))
        per_pixel_ce = per_pixel_ce * modulation
    return per_pixel_ce.mean()


def compute_identity_training_loss(
    identity_field: GaussianIdentityField,
    pixel_embeddings: torch.Tensor,
    target_labels: torch.Tensor,
    gaussian_xyz: torch.Tensor,
    *,
    ignore_index: int = -1,
    class_balance_power: float = 0.5,
    focal_gamma: float = 0.0,
    spatial_loss_weight: float = 0.1,
    spatial_k_neighbors: int = 8,
    spatial_max_samples: int | None = 4096,
) -> IdentityLossBreakdown:
    """Compute 2D supervision + 3D consistency losses.

    Args:
        identity_field: Learnable Gaussian identity field and classifier.
        pixel_embeddings: `(B, D, H, W)` rendered identity embeddings.
        target_labels: `(B, H, W)` scene-global class labels.
        gaussian_xyz: `(N, 3)` Gaussian centers.
        ignore_index: Label value ignored by cross-entropy.
        class_balance_power: Inverse-frequency weighting strength for rare classes.
        focal_gamma: Focal-loss gamma. Zero disables focal modulation.
        spatial_loss_weight: Weight on local 3D consistency.
        spatial_k_neighbors: Neighborhood size used for consistency.
        spatial_max_samples: Optional cap on Gaussian count used in the 3D regularizer.
    """

    if target_labels.ndim != 3:
        raise ValueError(
            f"Expected `(B, H, W)` target labels, got shape {tuple(target_labels.shape)}"
        )

    field_output = identity_field(pixel_embeddings)
    if field_output.pixel_logits.shape[0] != target_labels.shape[0] or field_output.pixel_logits.shape[2:] != target_labels.shape[1:]:
        raise ValueError(
            "Pixel logits shape does not align with target labels: "
            f"{tuple(field_output.pixel_logits.shape)} vs {tuple(target_labels.shape)}"
        )

    cross_entropy = balanced_focal_cross_entropy(
        field_output.pixel_logits,
        target_labels.long(),
        ignore_index=ignore_index,
        balance_power=class_balance_power,
        focal_gamma=focal_gamma,
    )
    spatial_consistency = identity_spatial_consistency_loss(
        gaussian_xyz,
        field_output.gaussian_embeddings,
        k_neighbors=spatial_k_neighbors,
        max_samples=spatial_max_samples,
    )
    total = cross_entropy + float(spatial_loss_weight) * spatial_consistency
    valid_pixel_count = int((target_labels != ignore_index).sum().item())
    return IdentityLossBreakdown(
        total=total,
        cross_entropy=cross_entropy,
        spatial_consistency=spatial_consistency,
        valid_pixel_count=valid_pixel_count,
    )
