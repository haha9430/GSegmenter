"""Ground-truth Gaussian grouping from InteriorGS object annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gsegmenter.data import InteriorGSObjectRecord


@dataclass(slots=True)
class InteriorGSBox:
    """Precomputed oriented box helper for one InteriorGS object.

    The InteriorGS label format stores eight 3D corners per object. We interpret
    them as a parallelepiped where corners `[0, 1, 3, 4]` define the three box
    edge directions. This lets us test Gaussian centers against the box in the
    same world frame as the scene PLY without assuming axis alignment.
    """

    object_id: int
    label: str
    corners: np.ndarray
    origin_xyz: np.ndarray
    transform_inv: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray
    volume: float


@dataclass(slots=True)
class InteriorGSGroup:
    """Editor-facing summary for one InteriorGS object group."""

    object_id: int
    label: str
    gaussian_count: int
    centroid_xyz: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray
    annotation_bbox_min_xyz: np.ndarray
    annotation_bbox_max_xyz: np.ndarray
    annotation_volume: float


def build_interiorgs_boxes(
    objects: tuple[InteriorGSObjectRecord, ...],
    *,
    start_object_id: int = 0,
    min_volume: float = 1e-8,
) -> list[InteriorGSBox]:
    """Convert InteriorGS object annotations into oriented box helpers.

    Args:
        objects: Parsed `labels.json` records.
        start_object_id: Fallback integer id base for objects without `instance_id`.
        min_volume: Reject numerically degenerate boxes below this volume.
    """

    boxes: list[InteriorGSBox] = []
    next_object_id = int(start_object_id)
    for record in objects:
        if record.bbox_corners is None:
            continue

        corners = np.asarray(record.bbox_corners, dtype=np.float32)
        if corners.shape != (8, 3):
            continue

        origin_xyz = corners[0]
        edge_x = corners[1] - origin_xyz
        edge_y = corners[3] - origin_xyz
        edge_z = corners[4] - origin_xyz
        basis = np.stack([edge_x, edge_y, edge_z], axis=1)
        det = float(np.linalg.det(basis))
        volume = abs(det)
        if not np.isfinite(volume) or volume <= min_volume:
            continue

        object_id = int(record.instance_id) if record.instance_id is not None else next_object_id
        next_object_id = max(next_object_id + 1, object_id + 1)
        boxes.append(
            InteriorGSBox(
                object_id=object_id,
                label=record.label,
                corners=corners,
                origin_xyz=origin_xyz,
                transform_inv=np.linalg.inv(basis).astype(np.float32),
                bbox_min_xyz=corners.min(axis=0),
                bbox_max_xyz=corners.max(axis=0),
                volume=volume,
            )
        )

    return boxes


def compute_points_in_box_mask(
    points_xyz: np.ndarray,
    box: InteriorGSBox,
    *,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Return a boolean mask for points inside one InteriorGS object box.

    Args:
        points_xyz: `(N, 3)` Gaussian centers in world coordinates.
        box: Precomputed oriented box.
        epsilon: Numerical margin for boundary points.
    """

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected `(N, 3)` points, got shape {points_xyz.shape}")

    local_coords = (points_xyz - box.origin_xyz) @ box.transform_inv.T
    return np.logical_and(
        np.all(local_coords >= -epsilon, axis=1),
        np.all(local_coords <= (1.0 + epsilon), axis=1),
    )


def assign_gaussians_to_interiorgs_objects(
    gaussian_xyz: np.ndarray,
    objects: tuple[InteriorGSObjectRecord, ...],
    *,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, list[InteriorGSBox]]:
    """Assign Gaussian centers to InteriorGS objects using annotation boxes.

    The assignment is deterministic: if multiple object boxes contain the same
    Gaussian center, the smallest annotation box wins. This favors specific
    furniture over broad structural envelopes when labels overlap.
    """

    gaussian_xyz = np.asarray(gaussian_xyz, dtype=np.float32)
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError(f"Expected `(N, 3)` Gaussian centers, got shape {gaussian_xyz.shape}")

    boxes = sorted(build_interiorgs_boxes(objects), key=lambda item: (item.volume, item.object_id))
    assignments = np.full((gaussian_xyz.shape[0],), -1, dtype=np.int32)
    assigned_mask = np.zeros((gaussian_xyz.shape[0],), dtype=bool)

    for box in boxes:
        inside_mask = compute_points_in_box_mask(gaussian_xyz, box, epsilon=epsilon)
        claim_mask = np.logical_and(inside_mask, ~assigned_mask)
        assignments[claim_mask] = box.object_id
        assigned_mask[claim_mask] = True

    return assignments, boxes


def summarize_interiorgs_groups(
    gaussian_object_ids: np.ndarray,
    gaussian_xyz: np.ndarray,
    boxes: list[InteriorGSBox],
) -> list[InteriorGSGroup]:
    """Build summaries for object groups assigned from InteriorGS labels."""

    box_lookup = {box.object_id: box for box in boxes}
    groups: list[InteriorGSGroup] = []
    for object_id in sorted(int(value) for value in np.unique(gaussian_object_ids) if value >= 0):
        gaussian_mask = gaussian_object_ids == object_id
        group_xyz = gaussian_xyz[gaussian_mask]
        box = box_lookup[object_id]
        groups.append(
            InteriorGSGroup(
                object_id=object_id,
                label=box.label,
                gaussian_count=int(group_xyz.shape[0]),
                centroid_xyz=group_xyz.mean(axis=0),
                bbox_min_xyz=group_xyz.min(axis=0),
                bbox_max_xyz=group_xyz.max(axis=0),
                annotation_bbox_min_xyz=box.bbox_min_xyz,
                annotation_bbox_max_xyz=box.bbox_max_xyz,
                annotation_volume=box.volume,
            )
        )
    return groups


def save_interiorgs_group_outputs(
    gaussian_object_ids: np.ndarray,
    groups: list[InteriorGSGroup],
    output_root: Path,
) -> None:
    """Persist InteriorGS object assignments in the same spirit as group outputs."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "gaussian_object_ids.npy", gaussian_object_ids.astype(np.int32))
    payload = {
        "group_count": len(groups),
        "assigned_gaussians": int(np.count_nonzero(gaussian_object_ids >= 0)),
        "unknown_gaussians": int(np.count_nonzero(gaussian_object_ids < 0)),
        "groups": [
            {
                "object_id": group.object_id,
                "label": group.label,
                "gaussian_count": group.gaussian_count,
                "centroid_xyz": group.centroid_xyz.tolist(),
                "bbox_min_xyz": group.bbox_min_xyz.tolist(),
                "bbox_max_xyz": group.bbox_max_xyz.tolist(),
                "annotation_bbox_min_xyz": group.annotation_bbox_min_xyz.tolist(),
                "annotation_bbox_max_xyz": group.annotation_bbox_max_xyz.tolist(),
                "annotation_volume": group.annotation_volume,
            }
            for group in groups
        ],
    }
    (output_root / "gaussian_groups.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
