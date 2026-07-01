from pathlib import Path

import numpy as np

from gsegmenter.mapping.association import LocalInstanceEvidence
from gsegmenter.mapping.object_proposals import (
    select_top_proposals,
    summarize_object_proposals,
    write_proposal_highlight_ply,
)
from gsegmenter.mapping.gaussian_io import load_gaussian_table


def test_summarize_object_proposals_keeps_majority_metadata() -> None:
    gaussian_object_ids = np.array([0, 0, 1, -1], dtype=np.int32)
    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [99.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    local_instances = [
        LocalInstanceEvidence(
            local_index=0,
            frame_index=0,
            instance_id=10,
            gaussian_indices=np.array([0, 1], dtype=np.int64),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            label="sofa couch",
            label_family="seat",
        ),
        LocalInstanceEvidence(
            local_index=1,
            frame_index=1,
            instance_id=10,
            gaussian_indices=np.array([2], dtype=np.int64),
            weights=np.array([0.7], dtype=np.float32),
            label="television tv",
            label_family="media",
        ),
    ]
    global_object_ids = np.array([0, 1], dtype=np.int32)

    proposals = summarize_object_proposals(
        gaussian_object_ids=gaussian_object_ids,
        gaussian_xyz=gaussian_xyz,
        local_instances=local_instances,
        global_object_ids=global_object_ids,
    )

    assert len(proposals) == 2
    assert proposals[0].label_family == "seat"
    assert proposals[0].gaussian_count == 2
    assert proposals[0].support_frame_count == 1
    assert np.allclose(proposals[0].centroid_xyz, np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_select_top_proposals_filters_by_size_and_frames() -> None:
    proposals = summarize_object_proposals(
        gaussian_object_ids=np.array([0, 0, 1, 2], dtype=np.int32),
        gaussian_xyz=np.zeros((4, 3), dtype=np.float32),
        local_instances=[
            LocalInstanceEvidence(0, 0, 0, np.array([0, 1]), np.ones(2), label_family="seat"),
            LocalInstanceEvidence(1, 1, 0, np.array([2]), np.ones(1), label_family="media"),
            LocalInstanceEvidence(2, 2, 0, np.array([3]), np.ones(1), label_family="table"),
        ],
        global_object_ids=np.array([0, 1, 2], dtype=np.int32),
    )

    assert select_top_proposals(proposals, limit=2, min_gaussians=2, min_support_frames=1) == {0}


def test_write_proposal_highlight_ply_preserves_gaussian_channels(tmp_path: Path) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
        ]
    )
    table = np.zeros((3,), dtype=dtype)
    header = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
        ("opacity", "float"),
    ]
    output_path = tmp_path / "highlighted.ply"

    write_proposal_highlight_ply(
        output_path=output_path,
        table=table,
        header_properties=header,
        proposal_ids=np.array([0, -1, 1], dtype=np.int32),
        keep_proposal_ids={0},
        dim_opacity_scale=0.5,
    )

    loaded, loaded_header = load_gaussian_table(output_path)

    assert loaded.shape[0] == 3
    assert loaded_header == header
    assert loaded["opacity"][0] == 0.0
    assert loaded["opacity"][1] < 0.0
    assert loaded["f_dc_0"][0] != 0.0
