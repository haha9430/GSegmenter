"""Diagnostics for inspecting grouped Gaussians across NerfStudio frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsegmenter.data.nerfstudio_scene import NerfstudioScene
from gsegmenter.render.projection import project_world_points


@dataclass(slots=True)
class GroupVisibilitySummary:
    """Visibility summary for a grouped object across the dataset."""

    object_id: int
    gaussian_count: int
    best_frame_index: int
    best_visible_count: int
    visibility_ratio: float


def select_group_ids(
    group_entries: list[dict[str, object]],
    *,
    top_k: int | None = None,
    skip_largest_n: int = 0,
    min_group_size: int = 0,
    include_object_ids: list[int] | None = None,
    exclude_object_ids: list[int] | None = None,
) -> list[int]:
    """Return filtered group ids in descending gaussian-count order."""

    groups = sorted(group_entries, key=lambda entry: int(entry["gaussian_count"]), reverse=True)
    if skip_largest_n > 0:
        groups = groups[skip_largest_n:]
    if min_group_size > 0:
        groups = [group for group in groups if int(group["gaussian_count"]) >= min_group_size]
    if include_object_ids is not None:
        included = {int(object_id) for object_id in include_object_ids}
        groups = [group for group in groups if int(group["global_object_id"]) in included]
    if exclude_object_ids is not None:
        excluded = {int(object_id) for object_id in exclude_object_ids}
        groups = [group for group in groups if int(group["global_object_id"]) not in excluded]
    if top_k is not None:
        groups = groups[:top_k]
    return [int(group["global_object_id"]) for group in groups]


def find_best_frame_for_group(
    scene: NerfstudioScene,
    points_world: np.ndarray,
    object_ids: np.ndarray,
    object_id: int,
) -> GroupVisibilitySummary:
    """Find the frame where a grouped object has the most visible Gaussians."""

    object_mask = object_ids == int(object_id)
    gaussian_count = int(np.count_nonzero(object_mask))
    if gaussian_count == 0:
        raise ValueError(f"Object id {object_id} does not match any Gaussian.")

    group_points = points_world[object_mask]
    best_frame_index = -1
    best_visible_count = -1
    for frame in scene.frames:
        projection = project_world_points(group_points, scene.intrinsics, frame)
        visible_count = int(np.count_nonzero(projection.valid_mask))
        if visible_count > best_visible_count:
            best_visible_count = visible_count
            best_frame_index = frame.index

    return GroupVisibilitySummary(
        object_id=int(object_id),
        gaussian_count=gaussian_count,
        best_frame_index=best_frame_index,
        best_visible_count=best_visible_count,
        visibility_ratio=float(best_visible_count / max(gaussian_count, 1)),
    )
