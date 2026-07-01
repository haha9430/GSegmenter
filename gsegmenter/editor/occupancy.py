"""Occupancy-aware placement checks for InteriorGS object editing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gsegmenter.data import InteriorGSObjectRecord, InteriorGSOccupancy


@dataclass(slots=True)
class OccupancyPlacementSummary:
    """Top-down occupancy evaluation for one edited object placement."""

    object_id: int
    label: str
    free_pixels: int
    occupied_pixels: int
    unknown_pixels: int
    outside_pixels: int
    occupied_fraction: float
    unknown_fraction: float
    valid: bool


def world_xy_to_occupancy_pixels(world_xy: np.ndarray, occupancy: InteriorGSOccupancy) -> np.ndarray:
    """Map world XY points into InteriorGS occupancy image coordinates.

    InteriorGS defines world axes as XYZ = (Right, Back, Up), while the
    occupancy PNG uses a top-down raster. The dataset's own helper converts
    points with:
        x_px = (-x_world + upper_x) / scale
        y_px = ( y_world - lower_y) / scale
    We mirror that exactly so editor constraints stay in the same frame as the
    provided occupancy map.
    """

    world_xy = np.asarray(world_xy, dtype=np.float32)
    if world_xy.ndim != 2 or world_xy.shape[1] != 2:
        raise ValueError(f"Expected `(N, 2)` XY points, got shape {world_xy.shape}")
    upper = np.asarray(occupancy.metadata["upper"], dtype=np.float32)
    lower = np.asarray(occupancy.metadata["lower"], dtype=np.float32)
    scale = float(occupancy.metadata["scale"])
    pixel_xy = np.empty_like(world_xy, dtype=np.float32)
    pixel_xy[:, 0] = (-world_xy[:, 0] + upper[0]) / scale
    pixel_xy[:, 1] = (+world_xy[:, 1] - lower[1]) / scale
    return pixel_xy


def _load_occupancy_image(occupancy: InteriorGSOccupancy) -> np.ndarray:
    return np.asarray(Image.open(Path(occupancy.image_path)).convert("L"), dtype=np.uint8)


def _transform_points(points_xyz: np.ndarray, rotation_matrix: np.ndarray, translation_xyz: np.ndarray, pivot_xyz: np.ndarray) -> np.ndarray:
    centered = points_xyz - pivot_xyz[None, :]
    rotated = centered @ rotation_matrix.T
    return rotated + pivot_xyz[None, :] + translation_xyz[None, :]


def evaluate_interiorgs_object_placement(
    record: InteriorGSObjectRecord,
    occupancy: InteriorGSOccupancy,
    *,
    object_id: int,
    translation_xyz: np.ndarray,
    rotation_matrix: np.ndarray,
    max_occupied_fraction: float = 0.05,
    max_unknown_fraction: float = 0.25,
) -> OccupancyPlacementSummary:
    """Evaluate whether a moved InteriorGS object lands in valid free space.

    The check uses the bottom face of the annotated 3D box as a top-down
    footprint. This is deliberate: for navigation-style placement constraints,
    floor contact matters more than the object's full vertical volume.
    """

    if record.bbox_corners is None:
        raise ValueError(f"Object {object_id} does not have annotation bbox corners.")

    bbox_corners = np.asarray(record.bbox_corners, dtype=np.float32)
    if bbox_corners.shape != (8, 3):
        raise ValueError(f"Expected `(8, 3)` bbox corners, got {bbox_corners.shape}")

    translation_xyz = np.asarray(translation_xyz, dtype=np.float32)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float32)
    if translation_xyz.shape != (3,):
        raise ValueError(f"Expected translation shape (3,), got {translation_xyz.shape}")
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation_matrix shape (3, 3), got {rotation_matrix.shape}")

    pivot_xyz = bbox_corners.mean(axis=0)
    bottom_face = bbox_corners[:4]
    transformed_bottom = _transform_points(bottom_face, rotation_matrix, translation_xyz, pivot_xyz)
    polygon_px = world_xy_to_occupancy_pixels(transformed_bottom[:, :2], occupancy)

    occupancy_img = _load_occupancy_image(occupancy)
    height, width = occupancy_img.shape
    mask_img = Image.new("L", (width, height), 0)
    polygon = [tuple(point.tolist()) for point in polygon_px]
    ImageDraw.Draw(mask_img).polygon(polygon, outline=255, fill=255)
    footprint_mask = np.asarray(mask_img, dtype=np.uint8) > 0
    total_pixels = int(np.count_nonzero(footprint_mask))
    if total_pixels == 0:
        return OccupancyPlacementSummary(
            object_id=object_id,
            label=record.label,
            free_pixels=0,
            occupied_pixels=0,
            unknown_pixels=0,
            outside_pixels=0,
            occupied_fraction=1.0,
            unknown_fraction=1.0,
            valid=False,
        )

    image_pixels = occupancy_img[footprint_mask]
    free_pixels = int(np.count_nonzero(image_pixels == 255))
    occupied_pixels = int(np.count_nonzero(image_pixels == 0))
    unknown_pixels = int(np.count_nonzero(image_pixels == 127))

    occupied_fraction = occupied_pixels / total_pixels
    unknown_fraction = unknown_pixels / total_pixels
    valid = occupied_fraction <= max_occupied_fraction and unknown_fraction <= max_unknown_fraction

    return OccupancyPlacementSummary(
        object_id=object_id,
        label=record.label,
        free_pixels=free_pixels,
        occupied_pixels=occupied_pixels,
        unknown_pixels=unknown_pixels,
        outside_pixels=0,
        occupied_fraction=float(occupied_fraction),
        unknown_fraction=float(unknown_fraction),
        valid=bool(valid),
    )
