"""Instance discovery from Grounded-SAM masks and COLMAP sparse tracks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gsegmenter.segmentation.mask_io import load_binary_mask


@dataclass(slots=True)
class TrackMaskEvidence:
    """Sparse COLMAP-track support for one frame-local mask."""

    local_index: int
    frame_stem: str
    source_instance_id: int
    category: str
    label: str
    mask_path: str
    score: float
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    point3d_ids: np.ndarray

    @property
    def support_size(self) -> int:
        return int(self.point3d_ids.size)


@dataclass(slots=True)
class TrackInstanceGroup:
    """One discovered image/COLMAP-based instance group."""

    group_id: int
    category: str
    mask_count: int
    total_track_observations: int
    unique_point_count: int
    member_local_indices: tuple[int, ...]


def collect_mask_track_ids(
    *,
    track_xy: np.ndarray,
    point3d_ids: np.ndarray,
    mask_path: Path,
    colmap_image_size: tuple[int, int],
    mask_image_size: tuple[int, int],
) -> np.ndarray:
    """Collect COLMAP 3D point ids whose 2D observations fall inside a mask."""

    if track_xy.ndim != 2 or track_xy.shape[1] != 2:
        raise ValueError(f"track_xy must have shape (N, 2), got {track_xy.shape}")
    if point3d_ids.shape != (track_xy.shape[0],):
        raise ValueError("point3d_ids length must match track_xy.")
    mask = load_binary_mask(mask_path)
    mask_width, mask_height = (int(value) for value in mask_image_size)
    if mask.shape != (mask_height, mask_width):
        raise ValueError(f"Mask {mask_path} has shape {mask.shape}, expected {(mask_height, mask_width)}")

    colmap_width, colmap_height = (int(value) for value in colmap_image_size)
    scale_x = mask_width / float(colmap_width)
    scale_y = mask_height / float(colmap_height)
    pixel_x = np.floor(track_xy[:, 0] * scale_x).astype(np.int64)
    pixel_y = np.floor(track_xy[:, 1] * scale_y).astype(np.int64)
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < mask_width)
        & (pixel_y >= 0)
        & (pixel_y < mask_height)
    )
    if not np.any(in_bounds):
        return np.zeros((0,), dtype=np.int64)
    hit = mask[pixel_y[in_bounds], pixel_x[in_bounds]]
    return np.unique(point3d_ids[in_bounds][hit]).astype(np.int64)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros((size,), dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        if parent != value:
            self.parent[value] = self.find(parent)
        return int(self.parent[value])

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def assign_track_instance_groups(
    evidences: list[TrackMaskEvidence],
    *,
    min_shared_points: int,
    min_overlap_ratio: float,
    min_group_masks: int,
) -> tuple[np.ndarray, list[TrackInstanceGroup]]:
    """Group masks that share enough COLMAP 3D points within each category."""

    if min_shared_points <= 0:
        raise ValueError("min_shared_points must be positive.")
    if min_overlap_ratio < 0.0 or min_overlap_ratio > 1.0:
        raise ValueError("min_overlap_ratio must be in [0, 1].")
    if min_group_masks <= 0:
        raise ValueError("min_group_masks must be positive.")
    if not evidences:
        return np.zeros((0,), dtype=np.int32), []

    union_find = _UnionFind(len(evidences))
    category_indices: dict[str, list[int]] = {}
    for index, evidence in enumerate(evidences):
        category_indices.setdefault(evidence.category, []).append(index)

    for indices in category_indices.values():
        point_to_masks: dict[int, list[int]] = {}
        for local_index in indices:
            for point_id in evidences[local_index].point3d_ids.tolist():
                point_to_masks.setdefault(int(point_id), []).append(local_index)
        shared_counts: dict[tuple[int, int], int] = {}
        for masks in point_to_masks.values():
            if len(masks) < 2:
                continue
            masks = sorted(set(masks))
            for lhs_position, lhs in enumerate(masks):
                for rhs in masks[lhs_position + 1 :]:
                    key = (lhs, rhs)
                    shared_counts[key] = shared_counts.get(key, 0) + 1
        for (lhs, rhs), shared in shared_counts.items():
            if shared < min_shared_points:
                continue
            denominator = float(min(evidences[lhs].support_size, evidences[rhs].support_size))
            if denominator <= 0.0:
                continue
            if shared / denominator >= min_overlap_ratio:
                union_find.union(lhs, rhs)

    root_members: dict[int, list[int]] = {}
    for index in range(len(evidences)):
        root_members.setdefault(union_find.find(index), []).append(index)

    sorted_groups = [
        members for members in root_members.values() if len(members) >= int(min_group_masks)
    ]
    sorted_groups.sort(
        key=lambda members: (
            len(members),
            sum(evidences[index].support_size for index in members),
        ),
        reverse=True,
    )

    group_ids = np.full((len(evidences),), -1, dtype=np.int32)
    groups: list[TrackInstanceGroup] = []
    for group_id, members in enumerate(sorted_groups):
        point_ids = np.concatenate([evidences[index].point3d_ids for index in members])
        for index in members:
            group_ids[index] = group_id
        groups.append(
            TrackInstanceGroup(
                group_id=group_id,
                category=evidences[members[0]].category,
                mask_count=len(members),
                total_track_observations=int(point_ids.size),
                unique_point_count=int(np.unique(point_ids).size),
                member_local_indices=tuple(int(index) for index in members),
            )
        )
    return group_ids, groups
