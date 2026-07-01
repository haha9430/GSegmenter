"""Multiview association from frame-local instances to global object hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class LocalInstanceEvidence:
    """Aggregated vote evidence for one frame-local instance."""

    local_index: int
    frame_index: int
    instance_id: int
    gaussian_indices: np.ndarray
    weights: np.ndarray
    label: str | None = None
    label_family: str | None = None

    @property
    def support_size(self) -> int:
        """Number of unique Gaussians supporting this local instance."""

        return int(self.gaussian_indices.size)


@dataclass(slots=True)
class AssociationPair:
    """Similarity edge between two frame-local instances."""

    lhs_local_index: int
    rhs_local_index: int
    shared_gaussians: int
    overlap_ratio: float


LABEL_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("seat", ("chair", "stool", "ottoman", "sofa", "couch")),
    ("table", ("table", "desk", "nightstand")),
    ("storage", ("cabinet", "shelf", "bookshelf", "wardrobe", "drawer", "dresser")),
    ("media", ("television", "tv", "speaker")),
    ("bed", ("bed",)),
    ("lighting", ("lamp",)),
)


def infer_label_family(label: str | None) -> str | None:
    """Map a noisy grounded detector phrase into a coarse furniture family."""

    if label is None:
        return None
    normalized = str(label).casefold()
    for family, keywords in LABEL_FAMILY_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return family
    return None


def load_vote_evidence(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sparse vote evidence arrays from disk."""

    payload = np.load(npz_path)
    return (
        payload["frame_indices"],
        payload["instance_ids"],
        payload["gaussian_indices"],
        payload["weights"],
    )


def aggregate_local_instances(
    frame_indices: np.ndarray,
    instance_ids: np.ndarray,
    gaussian_indices: np.ndarray,
    weights: np.ndarray,
) -> list[LocalInstanceEvidence]:
    """Collapse sparse vote rows into one record per frame-local instance."""

    if len(frame_indices) == 0:
        return []

    order = np.lexsort((gaussian_indices, instance_ids, frame_indices))
    frame_indices = frame_indices[order]
    instance_ids = instance_ids[order]
    gaussian_indices = gaussian_indices[order]
    weights = weights[order]

    local_instances: list[LocalInstanceEvidence] = []
    start = 0
    local_index = 0
    while start < len(frame_indices):
        frame_index = int(frame_indices[start])
        instance_id = int(instance_ids[start])
        end = start + 1
        while end < len(frame_indices):
            if int(frame_indices[end]) != frame_index or int(instance_ids[end]) != instance_id:
                break
            end += 1

        local_gaussians = gaussian_indices[start:end]
        local_weights = weights[start:end]
        unique_gaussians, inverse = np.unique(local_gaussians, return_inverse=True)
        summed_weights = np.zeros((len(unique_gaussians),), dtype=np.float32)
        np.add.at(summed_weights, inverse, local_weights.astype(np.float32))
        local_instances.append(
            LocalInstanceEvidence(
                local_index=local_index,
                frame_index=frame_index,
                instance_id=instance_id,
                gaussian_indices=unique_gaussians.astype(np.int64),
                weights=summed_weights,
            )
        )
        local_index += 1
        start = end

    return local_instances


def _intersection_size(lhs: np.ndarray, rhs: np.ndarray) -> int:
    """Compute the size of the intersection between sorted unique arrays."""

    lhs_ptr = 0
    rhs_ptr = 0
    shared = 0
    while lhs_ptr < len(lhs) and rhs_ptr < len(rhs):
        lhs_value = lhs[lhs_ptr]
        rhs_value = rhs[rhs_ptr]
        if lhs_value == rhs_value:
            shared += 1
            lhs_ptr += 1
            rhs_ptr += 1
        elif lhs_value < rhs_value:
            lhs_ptr += 1
        else:
            rhs_ptr += 1
    return shared


def build_association_pairs(
    local_instances: list[LocalInstanceEvidence],
    *,
    max_frame_gap: int = 1,
    min_shared_gaussians: int = 32,
    min_overlap_ratio: float = 0.1,
    require_same_label_family: bool = False,
) -> list[AssociationPair]:
    """Create graph edges between temporally nearby local instances."""

    frame_buckets: dict[int, list[LocalInstanceEvidence]] = {}
    for instance in local_instances:
        frame_buckets.setdefault(instance.frame_index, []).append(instance)

    pairs: list[AssociationPair] = []
    frame_indices = sorted(frame_buckets)
    for lhs_frame in frame_indices:
        lhs_instances = frame_buckets[lhs_frame]
        for frame_gap in range(1, max_frame_gap + 1):
            rhs_frame = lhs_frame + frame_gap
            rhs_instances = frame_buckets.get(rhs_frame)
            if rhs_instances is None:
                continue

            rhs_by_gaussian: dict[int, list[int]] = {}
            rhs_support_sizes: dict[int, int] = {}
            for rhs in rhs_instances:
                rhs_support_sizes[rhs.local_index] = rhs.support_size
                for gaussian_index in rhs.gaussian_indices.tolist():
                    rhs_by_gaussian.setdefault(int(gaussian_index), []).append(rhs.local_index)
            rhs_by_local_index = {rhs.local_index: rhs for rhs in rhs_instances}

            for lhs in lhs_instances:
                candidate_shared_counts: dict[int, int] = {}
                for gaussian_index in lhs.gaussian_indices.tolist():
                    rhs_candidates = rhs_by_gaussian.get(int(gaussian_index))
                    if rhs_candidates is None:
                        continue
                    for rhs_local_index in rhs_candidates:
                        candidate_shared_counts[rhs_local_index] = (
                            candidate_shared_counts.get(rhs_local_index, 0) + 1
                        )

                for rhs_local_index, shared in candidate_shared_counts.items():
                    rhs = rhs_by_local_index[rhs_local_index]
                    if require_same_label_family:
                        if lhs.label_family is None or rhs.label_family is None:
                            continue
                        if lhs.label_family != rhs.label_family:
                            continue
                    if shared < min_shared_gaussians:
                        continue
                    overlap_ratio = shared / float(
                        min(lhs.support_size, rhs_support_sizes[rhs_local_index])
                    )
                    if overlap_ratio < min_overlap_ratio:
                        continue
                    pairs.append(
                        AssociationPair(
                            lhs_local_index=lhs.local_index,
                            rhs_local_index=rhs_local_index,
                            shared_gaussians=shared,
                            overlap_ratio=overlap_ratio,
                        )
                    )
    return pairs


class _UnionFind:
    """Small union-find helper for graph components."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros((size,), dtype=np.int64)

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(int(parent))
        return int(self.parent[value])

    def union(self, lhs: int, rhs: int) -> None:
        lhs_root = self.find(lhs)
        rhs_root = self.find(rhs)
        if lhs_root == rhs_root:
            return
        if self.rank[lhs_root] < self.rank[rhs_root]:
            lhs_root, rhs_root = rhs_root, lhs_root
        self.parent[rhs_root] = lhs_root
        if self.rank[lhs_root] == self.rank[rhs_root]:
            self.rank[lhs_root] += 1


def assign_global_objects(
    local_instances: list[LocalInstanceEvidence],
    pairs: list[AssociationPair],
    *,
    total_local_count: int | None = None,
) -> np.ndarray:
    """Assign a global object id to every local instance via connected components."""

    output_size = int(total_local_count) if total_local_count is not None else len(local_instances)
    if not local_instances:
        return np.full((output_size,), -1, dtype=np.int32)

    union_find_size = max(output_size, max(instance.local_index for instance in local_instances) + 1)
    union_find = _UnionFind(union_find_size)
    for pair in pairs:
        union_find.union(pair.lhs_local_index, pair.rhs_local_index)

    active_indices = np.array([instance.local_index for instance in local_instances], dtype=np.int64)
    component_roots = np.array([union_find.find(int(index)) for index in active_indices])
    _, active_global_ids = np.unique(component_roots, return_inverse=True)
    global_ids = np.full((output_size,), -1, dtype=np.int32)
    global_ids[active_indices] = active_global_ids.astype(np.int32)
    return global_ids


def save_association_manifest(
    local_instances: list[LocalInstanceEvidence],
    global_object_ids: np.ndarray,
    pairs: list[AssociationPair],
    output_path: Path,
) -> None:
    """Persist association results in a JSON manifest."""

    payload = {
        "local_instance_count": len(local_instances),
        "global_object_count": int(np.unique(global_object_ids[global_object_ids >= 0]).size),
        "edge_count": len(pairs),
        "local_instances": [
            {
                "local_index": instance.local_index,
                "frame_index": instance.frame_index,
                "instance_id": instance.instance_id,
                "support_size": instance.support_size,
                "global_object_id": int(global_object_ids[instance.local_index]),
                "label": instance.label,
                "label_family": instance.label_family,
            }
            for instance in local_instances
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_association_edges(pairs: list[AssociationPair], output_path: Path) -> None:
    """Persist accepted graph edges as a compressed NPZ file."""

    if pairs:
        lhs = np.array([pair.lhs_local_index for pair in pairs], dtype=np.int32)
        rhs = np.array([pair.rhs_local_index for pair in pairs], dtype=np.int32)
        shared = np.array([pair.shared_gaussians for pair in pairs], dtype=np.int32)
        overlap = np.array([pair.overlap_ratio for pair in pairs], dtype=np.float32)
    else:
        lhs = np.zeros((0,), dtype=np.int32)
        rhs = np.zeros((0,), dtype=np.int32)
        shared = np.zeros((0,), dtype=np.int32)
        overlap = np.zeros((0,), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        lhs_local_index=lhs,
        rhs_local_index=rhs,
        shared_gaussians=shared,
        overlap_ratio=overlap,
    )
