from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import InteriorGSObjectRecord  # noqa: E402
from gsegmenter.mapping.interiorgs_grouping import (  # noqa: E402
    assign_gaussians_to_interiorgs_objects,
    build_interiorgs_boxes,
    compute_points_in_box_mask,
    summarize_interiorgs_groups,
)


def _make_box(origin_xyz: tuple[float, float, float], size_xyz: tuple[float, float, float]) -> np.ndarray:
    ox, oy, oz = origin_xyz
    sx, sy, sz = size_xyz
    return np.asarray(
        [
            [ox, oy, oz],
            [ox + sx, oy, oz],
            [ox + sx, oy + sy, oz],
            [ox, oy + sy, oz],
            [ox, oy, oz + sz],
            [ox + sx, oy, oz + sz],
            [ox + sx, oy + sy, oz + sz],
            [ox, oy + sy, oz + sz],
        ],
        dtype=np.float32,
    )


def test_build_interiorgs_boxes_uses_instance_ids():
    objects = (
        InteriorGSObjectRecord(
            instance_id=7,
            label="chair",
            bbox_corners=_make_box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            raw_payload={},
        ),
    )

    boxes = build_interiorgs_boxes(objects)

    assert len(boxes) == 1
    assert boxes[0].object_id == 7
    assert np.isclose(boxes[0].volume, 6.0)


def test_compute_points_in_box_mask_marks_inside_points():
    box_record = InteriorGSObjectRecord(
        instance_id=1,
        label="table",
        bbox_corners=_make_box((1.0, 2.0, 3.0), (2.0, 2.0, 1.0)),
        raw_payload={},
    )
    box = build_interiorgs_boxes((box_record,))[0]
    points = np.asarray(
        [
            [1.5, 2.5, 3.5],
            [3.0, 4.0, 4.0],
            [3.1, 4.0, 4.0],
            [0.5, 2.5, 3.5],
        ],
        dtype=np.float32,
    )

    inside = compute_points_in_box_mask(points, box)

    assert inside.tolist() == [True, True, False, False]


def test_assign_gaussians_to_interiorgs_objects_prefers_smaller_overlap():
    objects = (
        InteriorGSObjectRecord(
            instance_id=10,
            label="cabinet",
            bbox_corners=_make_box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)),
            raw_payload={},
        ),
        InteriorGSObjectRecord(
            instance_id=11,
            label="cup",
            bbox_corners=_make_box((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            raw_payload={},
        ),
    )
    xyz = np.asarray(
        [
            [1.2, 1.2, 1.2],
            [3.0, 3.0, 3.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )

    assignments, boxes = assign_gaussians_to_interiorgs_objects(xyz, objects)
    groups = summarize_interiorgs_groups(assignments, xyz, boxes)

    assert assignments.tolist() == [11, 10, -1]
    assert {group.object_id for group in groups} == {10, 11}
    assert {group.label for group in groups} == {"cabinet", "cup"}
