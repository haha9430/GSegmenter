"""Identity-field modules inspired by Gaussian Grouping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class IdentityFieldOutput:
    """Outputs produced by the identity rendering head.

    Shapes:
    - `gaussian_embeddings`: `(N, D)`
    - `pixel_embeddings`: `(B, D, H, W)`
    - `pixel_logits`: `(B, C, H, W)`
    """

    gaussian_embeddings: torch.Tensor
    pixel_embeddings: torch.Tensor
    pixel_logits: torch.Tensor


class GaussianIdentityField(nn.Module):
    """Learnable per-Gaussian identity embeddings with a 1x1 classifier head.

    This mirrors the Gaussian Grouping idea that object identity should be a
    trainable Gaussian attribute rather than a post-hoc clustering label.
    """

    def __init__(self, num_gaussians: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.gaussian_embeddings = nn.Parameter(
            torch.zeros((num_gaussians, embedding_dim), dtype=torch.float32)
        )
        nn.init.normal_(self.gaussian_embeddings, mean=0.0, std=0.01)
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, pixel_embeddings: torch.Tensor) -> IdentityFieldOutput:
        """Classify rendered identity embeddings into per-pixel logits."""

        if pixel_embeddings.ndim != 4:
            raise ValueError(
                f"Expected `(B, D, H, W)` pixel embeddings, got shape {tuple(pixel_embeddings.shape)}"
            )
        if pixel_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "Pixel embedding channel count does not match configured embedding_dim: "
                f"{pixel_embeddings.shape[1]} vs {self.embedding_dim}"
            )
        pixel_logits = self.classifier(pixel_embeddings)
        return IdentityFieldOutput(
            gaussian_embeddings=self.gaussian_embeddings,
            pixel_embeddings=pixel_embeddings,
            pixel_logits=pixel_logits,
        )
