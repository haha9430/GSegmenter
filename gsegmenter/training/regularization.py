"""Regularizers for identity-aware Gaussian training."""

from __future__ import annotations

import torch


def identity_spatial_consistency_loss(
    xyz: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    k_neighbors: int = 8,
    max_samples: int | None = 4096,
) -> torch.Tensor:
    """Encourage nearby Gaussians to carry similar identity embeddings.

    Args:
        xyz: `(N, 3)` Gaussian centers in world space.
        embeddings: `(N, D)` identity embedding vectors.
        k_neighbors: Number of nearest neighbors used for local consistency.
        max_samples: Optional cap on Gaussian count used for the regularizer.

    Returns:
        Scalar tensor suitable as a regularization term.
    """

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected `(N, 3)` xyz tensor, got shape {tuple(xyz.shape)}")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected `(N, D)` embeddings, got shape {tuple(embeddings.shape)}")
    if xyz.shape[0] != embeddings.shape[0]:
        raise ValueError("xyz and embeddings must share the same first dimension.")
    if xyz.shape[0] < 2:
        return embeddings.new_zeros(())

    if max_samples is not None and max_samples > 0 and xyz.shape[0] > int(max_samples):
        sample_indices = torch.randperm(xyz.shape[0], device=xyz.device)[: int(max_samples)]
        xyz = xyz[sample_indices]
        embeddings = embeddings[sample_indices]

    k_neighbors = max(1, min(int(k_neighbors), xyz.shape[0] - 1))
    distances = torch.cdist(xyz, xyz, p=2)
    neighbor_indices = torch.topk(distances, k=k_neighbors + 1, largest=False).indices[:, 1:]
    neighbor_embeddings = embeddings[neighbor_indices]
    center_embeddings = embeddings[:, None, :]
    squared_differences = (center_embeddings - neighbor_embeddings).pow(2).sum(dim=-1)
    return squared_differences.mean()
