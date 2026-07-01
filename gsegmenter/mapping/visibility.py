"""Visibility helpers for projection-based Gaussian scoring."""

from __future__ import annotations

import numpy as np


def build_front_depth_buffer(
    image_points: np.ndarray,
    depths: np.ndarray,
    valid_mask: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Build a per-pixel nearest-depth buffer from projected Gaussian centers.

    Args:
        image_points: Projected image coordinates as `(N, 2)` in pixel units.
        depths: Camera-space positive depths as `(N,)`; smaller means closer.
        valid_mask: `(N,)` mask for points inside the image and in front of the camera.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        `(height, width)` float32 array. Pixels with no projected Gaussian are `inf`.
    """

    image_points = np.asarray(image_points, dtype=np.float32)
    depths = np.asarray(depths, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError(f"image_points must have shape (N, 2), got {image_points.shape}")
    if depths.shape != (image_points.shape[0],):
        raise ValueError(f"depths must have shape ({image_points.shape[0]},), got {depths.shape}")
    if valid_mask.shape != (image_points.shape[0],):
        raise ValueError(f"valid_mask must have shape ({image_points.shape[0]},), got {valid_mask.shape}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")

    front_depth = np.full((height, width), np.inf, dtype=np.float32)
    indices = np.flatnonzero(valid_mask)
    if indices.size == 0:
        return front_depth

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
        return front_depth

    np.minimum.at(front_depth, (pixel_y, pixel_x), depths[indices])
    return front_depth


def filter_front_visible_points(
    image_points: np.ndarray,
    depths: np.ndarray,
    local_indices: np.ndarray,
    front_depth: np.ndarray,
    *,
    margin_ratio: float = 0.05,
    min_margin: float = 0.03,
) -> np.ndarray:
    """Keep local point indices near the front-most Gaussian depth at each pixel.

    This is an approximate center-based occlusion test. It prevents a cluster
    projected behind a visible surface from inheriting that foreground surface's
    2D mask label during semantic scoring.
    """

    if margin_ratio < 0.0:
        raise ValueError("margin_ratio must be non-negative.")
    if min_margin < 0.0:
        raise ValueError("min_margin must be non-negative.")
    if front_depth.ndim != 2:
        raise ValueError(f"front_depth must have shape (H, W), got {front_depth.shape}")

    local_indices = np.asarray(local_indices, dtype=np.int64)
    if local_indices.size == 0:
        return local_indices
    height, width = front_depth.shape
    pixel_x = np.floor(image_points[local_indices, 0]).astype(np.int64)
    pixel_y = np.floor(image_points[local_indices, 1]).astype(np.int64)
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    if not np.any(in_bounds):
        return np.zeros((0,), dtype=np.int64)

    bounded_indices = local_indices[in_bounds]
    pixel_x = pixel_x[in_bounds]
    pixel_y = pixel_y[in_bounds]
    nearest_depth = front_depth[pixel_y, pixel_x]
    finite = np.isfinite(nearest_depth)
    if not np.any(finite):
        return np.zeros((0,), dtype=np.int64)

    bounded_indices = bounded_indices[finite]
    nearest_depth = nearest_depth[finite]
    margin = np.maximum(float(min_margin), np.abs(nearest_depth) * float(margin_ratio))
    visible = depths[bounded_indices] <= (nearest_depth + margin)
    return bounded_indices[visible]
