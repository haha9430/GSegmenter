"""3D-first Gaussian cluster proposals built from spatial connectivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class GaussianClusterProposal:
    """Summary for one spatially connected Gaussian cluster."""

    global_object_id: int
    gaussian_count: int
    voxel_count: int
    centroid_xyz: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray

    @property
    def bbox_size_xyz(self) -> np.ndarray:
        """Return axis-aligned box side lengths in scene units."""

        return self.bbox_max_xyz - self.bbox_min_xyz

    @property
    def bbox_diag(self) -> float:
        """Return axis-aligned box diagonal length in scene units."""

        return float(np.linalg.norm(self.bbox_size_xyz))


class _UnionFind:
    """Small union-find helper for occupied voxel connectivity."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            self.parent[first_root] = second_root
        elif self.rank[first_root] > self.rank[second_root]:
            self.parent[second_root] = first_root
        else:
            self.parent[second_root] = first_root
            self.rank[first_root] += 1


def build_voxel_connected_components(
    xyz: np.ndarray,
    *,
    voxel_size: float,
    min_voxel_count: int = 1,
) -> np.ndarray:
    """Cluster Gaussian centers by 26-connected occupied voxels.

    Args:
        xyz: Gaussian centers as an `(N, 3)` array in one world coordinate frame.
        voxel_size: Edge length of a cubic voxel in the same scene units as `xyz`.
        min_voxel_count: Occupied voxels with fewer Gaussians are ignored before
            connectivity is computed.

    Returns:
        `(N,)` int32 array of raw connected component IDs. `-1` marks Gaussians
        in ignored sparse voxels. IDs are dense but not size-filtered.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")
    if min_voxel_count <= 0:
        raise ValueError("min_voxel_count must be positive.")
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz contains NaN or infinite values.")

    voxel_coords = np.floor(xyz / np.float32(voxel_size)).astype(np.int32)
    unique_voxels, inverse, voxel_counts = np.unique(
        voxel_coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    kept_voxel_mask = voxel_counts >= int(min_voxel_count)
    kept_indices = np.flatnonzero(kept_voxel_mask).astype(np.int32)
    if kept_indices.size == 0:
        return np.full((xyz.shape[0],), -1, dtype=np.int32)

    kept_coords = unique_voxels[kept_indices]
    compact_by_unique = np.full((unique_voxels.shape[0],), -1, dtype=np.int32)
    compact_by_unique[kept_indices] = np.arange(kept_indices.size, dtype=np.int32)

    coord_to_compact = {
        (int(coord[0]), int(coord[1]), int(coord[2])): int(index)
        for index, coord in enumerate(kept_coords)
    }
    positive_neighbor_offsets = [
        (dx, dy, dz)
        for dx in (0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz <= 0)
    ]

    union_find = _UnionFind(int(kept_indices.size))
    for index, coord in enumerate(kept_coords):
        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
        for dx, dy, dz in positive_neighbor_offsets:
            neighbor_index = coord_to_compact.get((x + dx, y + dy, z + dz))
            if neighbor_index is not None:
                union_find.union(index, neighbor_index)

    root_to_component: dict[int, int] = {}
    voxel_components = np.full((kept_indices.size,), -1, dtype=np.int32)
    for compact_index in range(kept_indices.size):
        root = union_find.find(compact_index)
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component)
        voxel_components[compact_index] = root_to_component[root]

    component_ids = np.full((xyz.shape[0],), -1, dtype=np.int32)
    compact_ids = compact_by_unique[inverse]
    valid_gaussians = compact_ids >= 0
    component_ids[valid_gaussians] = voxel_components[compact_ids[valid_gaussians]]
    return component_ids


def filter_and_remap_components(
    component_ids: np.ndarray,
    *,
    min_gaussians: int = 1,
    max_gaussians: int | None = None,
) -> np.ndarray:
    """Apply Gaussian-count thresholds and remap component IDs by descending size."""

    component_ids = np.asarray(component_ids, dtype=np.int32)
    if min_gaussians <= 0:
        raise ValueError("min_gaussians must be positive.")
    if max_gaussians is not None and max_gaussians < min_gaussians:
        raise ValueError("max_gaussians must be greater than or equal to min_gaussians.")

    positive_ids = component_ids[component_ids >= 0]
    if positive_ids.size == 0:
        return np.full(component_ids.shape, -1, dtype=np.int32)
    counts = np.bincount(positive_ids)
    keep_ids = [
        int(component_id)
        for component_id, count in enumerate(counts)
        if count >= min_gaussians and (max_gaussians is None or count <= max_gaussians)
    ]
    keep_ids.sort(key=lambda component_id: int(counts[component_id]), reverse=True)

    remap = np.full((counts.shape[0],), -1, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids):
        remap[old_id] = new_id
    filtered = np.full(component_ids.shape, -1, dtype=np.int32)
    valid = component_ids >= 0
    filtered[valid] = remap[component_ids[valid]]
    return filtered


def summarize_cluster_proposals(
    cluster_ids: np.ndarray,
    xyz: np.ndarray,
    *,
    voxel_size: float | None = None,
) -> list[GaussianClusterProposal]:
    """Summarize each non-negative cluster ID with count and bounding geometry."""

    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    xyz = np.asarray(xyz, dtype=np.float32)
    if cluster_ids.shape != (xyz.shape[0],):
        raise ValueError(
            f"cluster_ids must have shape ({xyz.shape[0]},), got {cluster_ids.shape}"
        )

    proposals: list[GaussianClusterProposal] = []
    for cluster_id in sorted(int(value) for value in np.unique(cluster_ids) if value >= 0):
        mask = cluster_ids == cluster_id
        cluster_xyz = xyz[mask]
        if voxel_size is None:
            voxel_count = int(cluster_xyz.shape[0])
        else:
            voxel_coords = np.floor(cluster_xyz / np.float32(voxel_size)).astype(np.int32)
            voxel_count = int(np.unique(voxel_coords, axis=0).shape[0])
        proposals.append(
            GaussianClusterProposal(
                global_object_id=cluster_id,
                gaussian_count=int(cluster_xyz.shape[0]),
                voxel_count=voxel_count,
                centroid_xyz=cluster_xyz.mean(axis=0),
                bbox_min_xyz=cluster_xyz.min(axis=0),
                bbox_max_xyz=cluster_xyz.max(axis=0),
            )
        )
    return proposals
