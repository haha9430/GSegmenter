"""Single-step training wrappers for identity-aware Gaussian supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gsegmenter.training.identity_bridge import IdentityTrainingBatch
from gsegmenter.training.identity_loss import IdentityLossBreakdown, compute_identity_training_loss
from gsegmenter.training.object_field import GaussianIdentityField


@dataclass(slots=True)
class IdentityStepResult:
    """Outputs and losses from one identity-aware optimization step."""

    losses: IdentityLossBreakdown
    batch_size: int


def run_identity_optimization_step(
    identity_field: GaussianIdentityField,
    batch: IdentityTrainingBatch,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    ignore_index: int = -1,
    spatial_loss_weight: float = 0.1,
    spatial_k_neighbors: int = 8,
    backward: bool = False,
) -> IdentityStepResult:
    """Run one supervised identity step and optionally update the optimizer."""

    losses = compute_identity_training_loss(
        identity_field,
        batch.pixel_embeddings,
        batch.target_labels,
        batch.gaussian_xyz,
        ignore_index=ignore_index,
        spatial_loss_weight=spatial_loss_weight,
        spatial_k_neighbors=spatial_k_neighbors,
    )
    if backward:
        losses.total.backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return IdentityStepResult(losses=losses, batch_size=batch.pixel_embeddings.shape[0])
