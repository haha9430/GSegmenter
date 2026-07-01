from __future__ import annotations

import numpy as np

from gsegmenter.mapping import (
    build_voxel_connected_components,
    filter_and_remap_components,
    summarize_cluster_proposals,
)


def test_voxel_connected_components_separates_spatial_clusters() -> None:
    xyz = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.08, 0.00, 0.00],
            [1.00, 1.00, 1.00],
            [1.04, 1.00, 1.00],
        ],
        dtype=np.float32,
    )

    component_ids = build_voxel_connected_components(xyz, voxel_size=0.05)

    assert component_ids.shape == (5,)
    assert component_ids[0] == component_ids[1] == component_ids[2]
    assert component_ids[3] == component_ids[4]
    assert component_ids[0] != component_ids[3]


def test_filter_and_remap_components_orders_by_size() -> None:
    component_ids = np.asarray([4, 4, 4, 2, 2, 1], dtype=np.int32)

    filtered = filter_and_remap_components(component_ids, min_gaussians=2)

    assert filtered.tolist() == [0, 0, 0, 1, 1, -1]


def test_summarize_cluster_proposals_reports_bounds() -> None:
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    cluster_ids = np.asarray([0, 0, -1], dtype=np.int32)

    proposals = summarize_cluster_proposals(cluster_ids, xyz, voxel_size=0.05)

    assert len(proposals) == 1
    assert proposals[0].global_object_id == 0
    assert proposals[0].gaussian_count == 2
    np.testing.assert_allclose(proposals[0].bbox_min_xyz, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(proposals[0].bbox_max_xyz, [0.1, 0.2, 0.0])
