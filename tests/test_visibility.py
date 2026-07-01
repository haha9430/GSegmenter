from __future__ import annotations

import numpy as np

from gsegmenter.mapping import build_front_depth_buffer, filter_front_visible_points


def test_build_front_depth_buffer_keeps_nearest_depth_per_pixel() -> None:
    image_points = np.asarray(
        [
            [1.2, 1.2],
            [1.4, 1.4],
            [2.2, 1.2],
        ],
        dtype=np.float32,
    )
    depths = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    valid_mask = np.asarray([True, True, True], dtype=bool)

    front_depth = build_front_depth_buffer(
        image_points,
        depths,
        valid_mask,
        height=4,
        width=4,
    )

    assert np.isclose(front_depth[1, 1], 1.0)
    assert np.isclose(front_depth[1, 2], 2.0)
    assert np.isinf(front_depth[0, 0])


def test_filter_front_visible_points_rejects_occluded_points() -> None:
    image_points = np.asarray(
        [
            [1.2, 1.2],
            [1.4, 1.4],
            [2.2, 1.2],
        ],
        dtype=np.float32,
    )
    depths = np.asarray([1.0, 1.04, 2.0], dtype=np.float32)
    front_depth = np.full((4, 4), np.inf, dtype=np.float32)
    front_depth[1, 1] = 1.0
    front_depth[1, 2] = 1.0
    local_indices = np.asarray([0, 1, 2], dtype=np.int64)

    kept = filter_front_visible_points(
        image_points,
        depths,
        local_indices,
        front_depth,
        margin_ratio=0.0,
        min_margin=0.05,
    )

    assert kept.tolist() == [0, 1]
