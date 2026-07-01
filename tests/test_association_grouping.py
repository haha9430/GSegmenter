from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.association import (  # noqa: E402
    aggregate_local_instances,
    assign_global_objects,
    build_association_pairs,
    infer_label_family,
)
from gsegmenter.mapping.grouping import (  # noqa: E402
    assign_gaussians_to_global_objects,
    summarize_gaussian_groups,
)


def test_aggregate_local_instances_merges_duplicate_votes():
    frame_indices = np.array([0, 0, 0, 1], dtype=np.int32)
    instance_ids = np.array([0, 0, 0, 1], dtype=np.int32)
    gaussian_indices = np.array([2, 2, 3, 3], dtype=np.int64)
    weights = np.array([0.5, 0.25, 1.0, 1.5], dtype=np.float32)

    local_instances = aggregate_local_instances(
        frame_indices, instance_ids, gaussian_indices, weights
    )

    assert len(local_instances) == 2
    assert local_instances[0].gaussian_indices.tolist() == [2, 3]
    assert np.allclose(local_instances[0].weights, np.array([0.75, 1.0], dtype=np.float32))


def test_build_association_pairs_links_overlapping_instances():
    frame_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    instance_ids = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    gaussian_indices = np.array([1, 2, 3, 2, 3, 4], dtype=np.int64)
    weights = np.ones((6,), dtype=np.float32)

    local_instances = aggregate_local_instances(
        frame_indices, instance_ids, gaussian_indices, weights
    )
    pairs = build_association_pairs(
        local_instances,
        max_frame_gap=1,
        min_shared_gaussians=2,
        min_overlap_ratio=0.5,
    )
    global_ids = assign_global_objects(local_instances, pairs)

    assert len(pairs) == 1
    assert global_ids.tolist() == [0, 0]


def test_build_association_pairs_can_require_matching_label_family():
    frame_indices = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    instance_ids = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    gaussian_indices = np.array([1, 2, 3, 2, 3, 4], dtype=np.int64)
    weights = np.ones((6,), dtype=np.float32)

    local_instances = aggregate_local_instances(
        frame_indices, instance_ids, gaussian_indices, weights
    )
    local_instances[0].label_family = "seat"
    local_instances[1].label_family = "media"

    pairs = build_association_pairs(
        local_instances,
        max_frame_gap=1,
        min_shared_gaussians=2,
        min_overlap_ratio=0.5,
        require_same_label_family=True,
    )

    assert pairs == []


def test_infer_label_family_maps_noisy_detector_phrases():
    assert infer_label_family("cabinet shelf bookshelf") == "storage"
    assert infer_label_family("television tv") == "media"
    assert infer_label_family("chair stool ottoman") == "seat"


def test_grouping_assigns_gaussians_to_global_object():
    frame_indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    instance_ids = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    gaussian_indices = np.array([0, 1, 0, 1, 3, 4], dtype=np.int64)
    weights = np.ones((6,), dtype=np.float32)

    local_instances = aggregate_local_instances(
        frame_indices, instance_ids, gaussian_indices, weights
    )
    pairs = build_association_pairs(
        local_instances,
        max_frame_gap=1,
        min_shared_gaussians=2,
        min_overlap_ratio=0.5,
    )
    global_ids = assign_global_objects(local_instances, pairs)
    gaussian_object_ids = assign_gaussians_to_global_objects(
        local_instances,
        global_ids,
        gaussian_count=5,
    )

    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    groups = summarize_gaussian_groups(gaussian_object_ids, xyz, local_instances, global_ids)

    assert gaussian_object_ids[0] == gaussian_object_ids[1]
    assert gaussian_object_ids[2] == -1
    assert len(groups) == 2
