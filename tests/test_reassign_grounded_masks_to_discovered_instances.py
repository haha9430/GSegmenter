from __future__ import annotations

import numpy as np

from tools.reassign_grounded_masks_to_discovered_instances import (
    _instance_class_name,
    _select_discovered_instances,
    choose_instance_for_mask,
)


def test_choose_instance_for_mask_uses_majority_allowed_instance() -> None:
    gaussian_instance_ids = np.asarray([0, 0, 1, 1, 1, -1], dtype=np.int32)

    chosen, hit_count, hit_ratio = choose_instance_for_mask(
        np.asarray([0, 2, 3, 4, 5], dtype=np.int64),
        gaussian_instance_ids,
        allowed_instance_ids={0, 1},
        min_hit_ratio=0.4,
        min_hit_count=2,
    )

    assert chosen == 1
    assert hit_count == 3
    assert hit_ratio == 0.6


def test_choose_instance_for_mask_rejects_weak_overlap() -> None:
    gaussian_instance_ids = np.asarray([0, 1, -1, -1], dtype=np.int32)

    chosen, hit_count, hit_ratio = choose_instance_for_mask(
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        gaussian_instance_ids,
        allowed_instance_ids={0, 1},
        min_hit_ratio=0.6,
        min_hit_count=1,
    )

    assert chosen is None
    assert hit_count == 1
    assert hit_ratio == 0.25


def test_select_discovered_instances_and_class_names() -> None:
    instances = [
        {"instance_id": 0, "category": "tv", "rank_in_category": 0},
        {"instance_id": 1, "category": "chair", "rank_in_category": 2},
    ]

    selected = _select_discovered_instances(instances, include_ids=None, exclude_ids={0})

    assert [instance["instance_id"] for instance in selected] == [1]
    assert _instance_class_name(selected[0]) == "chair_03"
