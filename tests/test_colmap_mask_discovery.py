from __future__ import annotations

from pathlib import Path

import numpy as np

from gsegmenter.mapping.colmap_mask_discovery import (
    TrackMaskEvidence,
    assign_track_instance_groups,
    collect_mask_track_ids,
)
from gsegmenter.segmentation.mask_io import save_binary_mask


def test_collect_mask_track_ids_scales_colmap_points_to_mask_size(tmp_path: Path) -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    mask_path = tmp_path / "mask.png"
    save_binary_mask(mask, mask_path)
    track_xy = np.asarray([[40.0, 40.0], [80.0, 80.0]], dtype=np.float32)
    point_ids = np.asarray([11, 22], dtype=np.int64)

    hits = collect_mask_track_ids(
        track_xy=track_xy,
        point3d_ids=point_ids,
        mask_path=mask_path,
        colmap_image_size=(200, 200),
        mask_image_size=(50, 50),
    )

    assert hits.tolist() == [11]


def test_assign_track_instance_groups_connects_shared_points_within_category() -> None:
    evidences = [
        TrackMaskEvidence(0, "a", 0, "chair", "chair", "m0.png", 0.9, 10, (0, 0, 1, 1), np.array([1, 2, 3])),
        TrackMaskEvidence(1, "b", 0, "chair", "chair", "m0.png", 0.9, 10, (0, 0, 1, 1), np.array([2, 3, 4])),
        TrackMaskEvidence(2, "c", 0, "table", "table", "m0.png", 0.9, 10, (0, 0, 1, 1), np.array([2, 3, 4])),
    ]

    group_ids, groups = assign_track_instance_groups(
        evidences,
        min_shared_points=2,
        min_overlap_ratio=0.5,
        min_group_masks=2,
    )

    assert group_ids.tolist() == [0, 0, -1]
    assert len(groups) == 1
    assert groups[0].category == "chair"
