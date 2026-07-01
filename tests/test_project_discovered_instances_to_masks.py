from __future__ import annotations

import numpy as np

from tools.project_discovered_instances_to_masks import (
    _instance_class_name,
    _select_instances,
    rasterize_projected_points,
)


def test_instance_class_name_uses_category_and_rank() -> None:
    assert _instance_class_name({"category": "chair", "rank_in_category": 1, "instance_id": 7}) == "chair_02"


def test_select_instances_filters_ids() -> None:
    instances = [
        {"instance_id": 0, "category": "tv"},
        {"instance_id": 1, "category": "chair"},
        {"instance_id": 2, "category": "table"},
    ]

    selected = _select_instances(instances, include_ids={0, 1, 2}, exclude_ids={1})

    assert [instance["instance_id"] for instance in selected] == [0, 2]


def test_rasterize_projected_points_dilates_seed_pixels() -> None:
    image_points = np.asarray([[2.0, 2.0], [6.0, 6.0]], dtype=np.float32)
    gaussian_mask = np.asarray([True, False])

    mask = rasterize_projected_points(
        image_points,
        gaussian_mask,
        image_shape=(8, 8),
        dilation_radius=1,
    )

    assert mask[2, 2]
    assert int(mask.sum()) == 9
