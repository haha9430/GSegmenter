"""Final Gaussian grouping from global object hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gsegmenter.mapping.association import LocalInstanceEvidence


@dataclass(slots=True)
class GaussianGroup:
    """Editor-facing summary for one grouped object."""

    global_object_id: int
    gaussian_count: int
    centroid_xyz: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray
    support_frames: np.ndarray
    total_vote_weight: float


def assign_gaussians_to_global_objects(
    local_instances: list[LocalInstanceEvidence],
    global_object_ids: np.ndarray,
    gaussian_count: int,
    *,
    min_vote_weight: float = 1e-4,
) -> np.ndarray:
    """Assign each Gaussian to the strongest global object hypothesis.

    Returns:
        `(N,)` int32 array where `-1` marks unknown/background.
    """

    if gaussian_count <= 0:
        return np.zeros((0,), dtype=np.int32)
    if not local_instances:
        return np.full((gaussian_count,), -1, dtype=np.int32)

    active_global_ids = global_object_ids[global_object_ids >= 0]
    global_count = int(active_global_ids.max() + 1) if len(active_global_ids) else 0
    vote_table = np.zeros((gaussian_count, global_count), dtype=np.float32)
    for instance in local_instances:
        global_id = int(global_object_ids[instance.local_index])
        if global_id < 0:
            continue
        vote_table[instance.gaussian_indices, global_id] += instance.weights
    if global_count == 0:
        return np.full((gaussian_count,), -1, dtype=np.int32)

    best_ids = np.argmax(vote_table, axis=1).astype(np.int32)
    best_weights = vote_table[np.arange(gaussian_count), best_ids]
    best_ids[best_weights <= min_vote_weight] = -1
    return best_ids


def summarize_gaussian_groups(
    gaussian_object_ids: np.ndarray,
    gaussian_xyz: np.ndarray,
    local_instances: list[LocalInstanceEvidence],
    global_object_ids: np.ndarray,
) -> list[GaussianGroup]:
    """Build editor-facing summaries for each global object."""

    groups: list[GaussianGroup] = []
    positive_ids = sorted(int(value) for value in np.unique(gaussian_object_ids) if value >= 0)
    for global_id in positive_ids:
        gaussian_mask = gaussian_object_ids == global_id
        group_xyz = gaussian_xyz[gaussian_mask]
        support_frames = sorted(
            {
                instance.frame_index
                for instance in local_instances
                if int(global_object_ids[instance.local_index]) == global_id
            }
        )
        total_vote_weight = float(
            sum(
                float(instance.weights.sum())
                for instance in local_instances
                if int(global_object_ids[instance.local_index]) == global_id
            )
        )
        groups.append(
            GaussianGroup(
                global_object_id=global_id,
                gaussian_count=int(group_xyz.shape[0]),
                centroid_xyz=group_xyz.mean(axis=0),
                bbox_min_xyz=group_xyz.min(axis=0),
                bbox_max_xyz=group_xyz.max(axis=0),
                support_frames=np.asarray(support_frames, dtype=np.int32),
                total_vote_weight=total_vote_weight,
            )
        )
    return groups


def save_gaussian_group_outputs(
    gaussian_object_ids: np.ndarray,
    groups: list[GaussianGroup],
    output_root: Path,
) -> None:
    """Persist the final Gaussian group assignments and summaries."""

    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "gaussian_object_ids.npy", gaussian_object_ids.astype(np.int32))
    payload = {
        "group_count": len(groups),
        "groups": [
            {
                "global_object_id": group.global_object_id,
                "gaussian_count": group.gaussian_count,
                "centroid_xyz": group.centroid_xyz.tolist(),
                "bbox_min_xyz": group.bbox_min_xyz.tolist(),
                "bbox_max_xyz": group.bbox_max_xyz.tolist(),
                "support_frames": group.support_frames.tolist(),
                "total_vote_weight": group.total_vote_weight,
            }
            for group in groups
        ],
    }
    (output_root / "gaussian_groups.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
