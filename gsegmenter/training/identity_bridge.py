"""Adapters between renderer outputs and identity-aware training code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gsegmenter.training.identity_dataset import SceneIdentityLabelFrame


@dataclass(slots=True)
class IdentityTrainingBatch:
    """Mini-batch inputs for a Gaussian Grouping style identity step.

    Shapes:
    - `pixel_embeddings`: `(B, D, H, W)`
    - `target_labels`: `(B, H, W)`
    - `gaussian_xyz`: `(N, 3)`
    """

    frame_indices: tuple[int, ...]
    pixel_embeddings: torch.Tensor
    target_labels: torch.Tensor
    gaussian_xyz: torch.Tensor


def extract_rendered_identity_embeddings(
    renderer_outputs: dict[str, torch.Tensor],
    *,
    render_key: str = "render_object",
) -> torch.Tensor:
    """Extract batched identity embeddings from a renderer output dictionary."""

    if render_key not in renderer_outputs:
        raise KeyError(
            f"Renderer outputs do not contain {render_key!r}. "
            "The identity-aware training path expects the renderer to return a "
            "per-pixel object embedding tensor under this key."
        )
    pixel_embeddings = renderer_outputs[render_key]
    if pixel_embeddings.ndim == 3:
        pixel_embeddings = pixel_embeddings.unsqueeze(0)
    if pixel_embeddings.ndim != 4:
        raise ValueError(
            f"Expected renderer embeddings with shape `(B, D, H, W)`, got {tuple(pixel_embeddings.shape)}"
        )
    return pixel_embeddings


def gather_scene_identity_targets(
    scene_frames: tuple[SceneIdentityLabelFrame, ...],
    frame_indices: list[int] | tuple[int, ...],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack scene-global label maps for a set of frame indices."""

    frame_lookup = {frame.frame_index: frame for frame in scene_frames}
    label_maps: list[torch.Tensor] = []
    for frame_index in frame_indices:
        if int(frame_index) not in frame_lookup:
            raise KeyError(f"Frame index {frame_index} is missing from scene identity supervision.")
        label_map = frame_lookup[int(frame_index)].label_map
        label_maps.append(torch.from_numpy(np.asarray(label_map, dtype=np.int64)))
    stacked = torch.stack(label_maps, dim=0)
    if device is not None:
        stacked = stacked.to(device)
    return stacked


def build_identity_training_batch(
    renderer_outputs: dict[str, torch.Tensor],
    scene_frames: tuple[SceneIdentityLabelFrame, ...],
    frame_indices: list[int] | tuple[int, ...],
    gaussian_xyz: torch.Tensor,
    *,
    render_key: str = "render_object",
) -> IdentityTrainingBatch:
    """Bundle renderer outputs and scene labels into a training batch."""

    pixel_embeddings = extract_rendered_identity_embeddings(renderer_outputs, render_key=render_key)
    target_labels = gather_scene_identity_targets(
        scene_frames,
        frame_indices,
        device=pixel_embeddings.device,
    )
    if pixel_embeddings.shape[0] != target_labels.shape[0]:
        raise ValueError(
            f"Batch size mismatch between rendered embeddings and labels: "
            f"{pixel_embeddings.shape[0]} vs {target_labels.shape[0]}"
        )
    return IdentityTrainingBatch(
        frame_indices=tuple(int(index) for index in frame_indices),
        pixel_embeddings=pixel_embeddings,
        target_labels=target_labels,
        gaussian_xyz=gaussian_xyz,
    )
