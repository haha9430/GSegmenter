from __future__ import annotations

from pathlib import Path

import numpy as np

from gsegmenter.mapping.gaussian_io import load_gaussian_table, write_gaussian_table
from gsegmenter.mapping.identity_instances import (
    build_identity_instance_ids,
    write_identity_instance_highlight_ply,
)


def test_build_identity_instance_ids_splits_components_within_class() -> None:
    xyz = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [1.04, 0.00, 0.00],
            [5.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    identity_ids = np.asarray([1, 1, 1, 1, 2], dtype=np.int32)
    probabilities = np.asarray(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )

    instance_ids, proposals = build_identity_instance_ids(
        xyz=xyz,
        identity_ids=identity_ids,
        probabilities=probabilities,
        class_names=["background", "chair", "table"],
        include_class_ids={1},
        min_confidence=0.85,
        voxel_size=0.05,
        min_gaussians=2,
    )

    assert instance_ids.tolist() == [0, 0, 1, 1, -1]
    assert [proposal.class_name for proposal in proposals] == ["chair", "chair"]
    assert [proposal.gaussian_count for proposal in proposals] == [2, 2]


def test_write_identity_instance_highlight_ply_preserves_schema(tmp_path: Path) -> None:
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
    source_path = tmp_path / "source.ply"
    output_path = tmp_path / "instances.ply"
    write_gaussian_table(source_path, table, header)
    loaded, loaded_header = load_gaussian_table(source_path)

    write_identity_instance_highlight_ply(
        output_path=output_path,
        table=loaded,
        header_properties=loaded_header,
        instance_ids=np.asarray([0, -1, 1], dtype=np.int32),
        dim_opacity_scale=0.5,
    )
    highlighted, highlighted_header = load_gaussian_table(output_path)

    assert highlighted.shape[0] == 3
    assert highlighted_header == header
    assert highlighted["f_dc_0"][0] != highlighted["f_dc_0"][2]
    assert highlighted["opacity"][1] < highlighted["opacity"][0]
