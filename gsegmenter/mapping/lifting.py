"""Lift per-frame 2D masks into sparse Gaussian vote evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gsegmenter.data.nerfstudio_scene import CameraIntrinsics, FrameRecord
from gsegmenter.render.projection import project_world_points
from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    load_binary_mask,
    load_frame_masks_manifest,
)


@dataclass(slots=True)
class VoteEvidence:
    """Sparse vote support for a local 2D instance."""

    frame_index: int
    instance_id: int
    gaussian_indices: np.ndarray
    weights: np.ndarray


def collect_mask_hits(
    image_points: np.ndarray,
    valid_mask: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Return Gaussian indices whose projected centers land inside the mask."""

    indices = np.nonzero(valid_mask)[0]
    if len(indices) == 0:
        return np.zeros((0,), dtype=np.int64)

    pixel_x = np.floor(image_points[indices, 0]).astype(np.int64)
    pixel_y = np.floor(image_points[indices, 1]).astype(np.int64)
    height, width = binary_mask.shape
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    if not np.any(in_bounds):
        return np.zeros((0,), dtype=np.int64)

    indices = indices[in_bounds]
    pixel_x = pixel_x[in_bounds]
    pixel_y = pixel_y[in_bounds]
    hit_mask = binary_mask[pixel_y, pixel_x]
    return indices[hit_mask]


def _fit_affine_robust(predicted: np.ndarray, target: np.ndarray, trim_quantile: float) -> tuple[float, float]:
    """Fit `target ~= scale * predicted + shift` with residual trimming."""

    finite = np.isfinite(predicted) & np.isfinite(target)
    x = predicted[finite].astype(np.float64)
    y = target[finite].astype(np.float64)
    if x.size < 32:
        return 1.0, 0.0

    keep = np.ones_like(x, dtype=bool)
    scale = 1.0
    shift = 0.0
    trim_quantile = float(np.clip(trim_quantile, 0.1, 1.0))
    for _ in range(4):
        design = np.stack([x[keep], np.ones(int(np.count_nonzero(keep)))], axis=1)
        scale, shift = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = np.abs((scale * x + shift) - y)
        cutoff = float(np.quantile(residual, trim_quantile))
        keep = residual <= cutoff
        if np.count_nonzero(keep) < 32:
            break
    return float(scale), float(shift)


def _front_surface_samples(
    image_points: np.ndarray,
    depths: np.ndarray,
    valid_mask: np.ndarray,
    depth_map: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample front-most projected Gaussian depth per occupied pixel for fitting."""

    indices = np.flatnonzero(valid_mask)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pixel_x = np.floor(image_points[indices, 0]).astype(np.int64)
    pixel_y = np.floor(image_points[indices, 1]).astype(np.int64)
    height, width = depth_map.shape
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    indices = indices[in_bounds]
    pixel_x = pixel_x[in_bounds]
    pixel_y = pixel_y[in_bounds]
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    linear_pixels = pixel_y * width + pixel_x
    order = np.lexsort((depths[indices], linear_pixels))
    sorted_linear = linear_pixels[order]
    sorted_indices = indices[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    front_indices = sorted_indices[first]
    front_x = pixel_x[order][first]
    front_y = pixel_y[order][first]

    if front_indices.size > max_points:
        rng = np.random.default_rng(42)
        chosen = rng.choice(front_indices.size, size=max_points, replace=False)
        front_indices = front_indices[chosen]
        front_x = front_x[chosen]
        front_y = front_y[chosen]

    return depth_map[front_y, front_x].astype(np.float32), depths[front_indices].astype(np.float32)


def build_depth_consistency_mask(
    image_points: np.ndarray,
    depths: np.ndarray,
    valid_mask: np.ndarray,
    depth_map: np.ndarray,
    *,
    fit_max_points: int = 30000,
    trim_quantile: float = 0.80,
    behind_margin_ratio: float = 0.18,
    behind_min_margin: float = 0.05,
) -> np.ndarray:
    """Return a frame-local mask that rejects points behind monocular surface depth.

    The monocular depth map is frame-locally affine aligned to front-most
    projected Gaussian depths. This does not estimate a room boundary; it only
    rejects Gaussians that project behind the visible surface for this view.
    """

    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth_map.shape}")

    fit_pred, fit_target = _front_surface_samples(
        image_points,
        depths,
        valid_mask,
        depth_map,
        int(fit_max_points),
    )
    scale, shift = _fit_affine_robust(fit_pred, fit_target, float(trim_quantile))

    depth_consistent = np.zeros_like(valid_mask, dtype=bool)
    indices = np.flatnonzero(valid_mask)
    if indices.size == 0:
        return depth_consistent
    pixel_x = np.floor(image_points[indices, 0]).astype(np.int64)
    pixel_y = np.floor(image_points[indices, 1]).astype(np.int64)
    height, width = depth_map.shape
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    indices = indices[in_bounds]
    pixel_x = pixel_x[in_bounds]
    pixel_y = pixel_y[in_bounds]
    if indices.size == 0:
        return depth_consistent

    aligned_depth = scale * depth_map[pixel_y, pixel_x].astype(np.float32) + shift
    margin = np.maximum(
        float(behind_min_margin),
        np.abs(aligned_depth) * float(behind_margin_ratio),
    )
    depth_consistent[indices] = depths[indices] <= (aligned_depth + margin)
    return depth_consistent


def build_front_surface_mask(
    image_points: np.ndarray,
    depths: np.ndarray,
    valid_mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    depth_margin: float = 0.03,
) -> np.ndarray:
    """Keep only Gaussians close to the front-most projected depth per pixel.

    Args:
        image_points: `(N, 2)` projected pixel coordinates.
        depths: `(N,)` camera-space positive depths in the same frame.
        valid_mask: `(N,)` projection validity mask.
        image_shape: `(height, width)` of the target image.
        depth_margin: Absolute camera-depth tolerance behind the front surface.

    This is a z-buffer style visibility approximation. It prevents mask votes
    from leaking through an object onto walls or other Gaussians behind the
    visible surface in the same pixel.
    """

    if depth_margin < 0.0:
        raise ValueError("depth_margin must be non-negative.")
    height, width = (int(value) for value in image_shape)
    front_surface = np.zeros_like(valid_mask, dtype=bool)
    indices = np.flatnonzero(valid_mask)
    if indices.size == 0:
        return front_surface

    pixel_x = np.floor(image_points[indices, 0]).astype(np.int64)
    pixel_y = np.floor(image_points[indices, 1]).astype(np.int64)
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    indices = indices[in_bounds]
    pixel_x = pixel_x[in_bounds]
    pixel_y = pixel_y[in_bounds]
    if indices.size == 0:
        return front_surface

    linear_pixels = pixel_y * width + pixel_x
    min_depth_by_pixel = np.full((height * width,), np.inf, dtype=np.float32)
    np.minimum.at(min_depth_by_pixel, linear_pixels, depths[indices].astype(np.float32))
    front_surface[indices] = depths[indices] <= (min_depth_by_pixel[linear_pixels] + float(depth_margin))
    return front_surface


def build_frame_vote_evidence(
    gaussian_xyz: np.ndarray,
    intrinsics: CameraIntrinsics,
    frame: FrameRecord,
    manifest: FrameMasksManifest,
    frame_dir: Path,
    opacity_weights: np.ndarray | None = None,
    quality_weights: np.ndarray | None = None,
    gaussian_valid_mask: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
    depth_fit_max_points: int = 30000,
    depth_trim_quantile: float = 0.80,
    depth_behind_margin_ratio: float = 0.18,
    depth_behind_min_margin: float = 0.05,
    front_surface_only: bool = False,
    front_surface_depth_margin: float = 0.03,
) -> list[VoteEvidence]:
    """Project Gaussians into one frame and collect sparse mask hits."""

    projection = project_world_points(gaussian_xyz, intrinsics, frame)
    projection_valid_mask = projection.valid_mask
    if front_surface_only:
        projection_valid_mask = projection_valid_mask & build_front_surface_mask(
            projection.image_points,
            projection.depths,
            projection.valid_mask,
            (intrinsics.height, intrinsics.width),
            depth_margin=front_surface_depth_margin,
        )
    if depth_map is not None:
        if depth_map.shape != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"Depth map shape {depth_map.shape} does not match frame image size "
                f"{(intrinsics.height, intrinsics.width)}"
            )
        projection_valid_mask = projection_valid_mask & build_depth_consistency_mask(
            projection.image_points,
            projection.depths,
            projection.valid_mask,
            depth_map,
            fit_max_points=depth_fit_max_points,
            trim_quantile=depth_trim_quantile,
            behind_margin_ratio=depth_behind_margin_ratio,
            behind_min_margin=depth_behind_min_margin,
        )
    if gaussian_valid_mask is not None:
        if gaussian_valid_mask.shape[0] != gaussian_xyz.shape[0]:
            raise ValueError(
                f"gaussian_valid_mask length {gaussian_valid_mask.shape[0]} does not match "
                f"gaussian count {gaussian_xyz.shape[0]}"
            )
        projection_valid_mask = projection_valid_mask & gaussian_valid_mask.astype(bool)
    if quality_weights is not None and quality_weights.shape[0] != gaussian_xyz.shape[0]:
        raise ValueError(
            f"quality_weights length {quality_weights.shape[0]} does not match gaussian count {gaussian_xyz.shape[0]}"
        )
    evidences: list[VoteEvidence] = []

    for instance in manifest.instances:
        mask_path = frame_dir / instance.mask_path
        binary_mask = load_binary_mask(mask_path)
        hit_indices = collect_mask_hits(
            projection.image_points,
            projection_valid_mask,
            binary_mask,
        )
        if len(hit_indices) == 0:
            continue

        weights = np.full((len(hit_indices),), instance.score, dtype=np.float32)
        if opacity_weights is not None:
            weights *= opacity_weights[hit_indices]
        if quality_weights is not None:
            weights *= quality_weights[hit_indices]

        evidences.append(
            VoteEvidence(
                frame_index=manifest.frame_index,
                instance_id=instance.instance_id,
                gaussian_indices=hit_indices.astype(np.int64),
                weights=weights.astype(np.float32),
            )
        )

    return evidences


def save_vote_evidence(evidences: list[VoteEvidence], output_path: Path) -> None:
    """Persist sparse vote evidence as a compressed NPZ bundle."""

    if evidences:
        frame_indices = np.concatenate(
            [
                np.full((len(evidence.gaussian_indices),), evidence.frame_index, dtype=np.int32)
                for evidence in evidences
            ]
        )
        instance_ids = np.concatenate(
            [
                np.full((len(evidence.gaussian_indices),), evidence.instance_id, dtype=np.int32)
                for evidence in evidences
            ]
        )
        gaussian_indices = np.concatenate([evidence.gaussian_indices for evidence in evidences])
        weights = np.concatenate([evidence.weights for evidence in evidences])
    else:
        frame_indices = np.zeros((0,), dtype=np.int32)
        instance_ids = np.zeros((0,), dtype=np.int32)
        gaussian_indices = np.zeros((0,), dtype=np.int64)
        weights = np.zeros((0,), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frame_indices=frame_indices,
        instance_ids=instance_ids,
        gaussian_indices=gaussian_indices,
        weights=weights,
    )


def save_vote_summary(
    evidences: list[VoteEvidence],
    output_path: Path,
    total_gaussians: int,
) -> None:
    """Write a lightweight JSON summary for quick inspection."""

    covered = 0
    if evidences:
        covered = int(np.unique(np.concatenate([evidence.gaussian_indices for evidence in evidences])).size)

    payload = {
        "frame_instance_count": len(evidences),
        "covered_gaussians": covered,
        "coverage_ratio": 0.0 if total_gaussians == 0 else covered / total_gaussians,
        "vote_count": int(sum(len(evidence.gaussian_indices) for evidence in evidences)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_frame_manifest_from_dir(frame_dir: Path) -> FrameMasksManifest:
    """Load the standard `instances.json` manifest from a frame output directory."""

    return load_frame_masks_manifest(frame_dir / "instances.json")
