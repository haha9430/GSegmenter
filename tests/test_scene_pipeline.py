from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.data.nerfstudio_scene import load_nerfstudio_scene
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud
from gsegmenter.render.projection import project_world_points


def _write_test_scene(root: Path) -> Path:
    dataset_root = root / "scene"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame_00001.png").write_bytes(b"")

    transforms = {
        "fl_x": 100.0,
        "fl_y": 100.0,
        "cx": 50.0,
        "cy": 50.0,
        "w": 100,
        "h": 100,
        "camera_model": "PINHOLE",
        "frames": [
            {
                "file_path": "images/frame_00001.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (dataset_root / "transforms.json").write_text(
        json.dumps(transforms),
        encoding="utf-8",
    )
    return dataset_root


def _write_test_ply(path: Path) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")
    row_format = "<11f"
    rows = [
        (0.0, 0.0, 2.0, 0.1, 1.0, 1.1, 1.2, 1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 2.0, 0.2, 2.0, 2.1, 2.2, 0.0, 1.0, 0.0, 0.0),
    ]

    with path.open("wb") as handle:
        handle.write(header)
        for row in rows:
            handle.write(struct.pack(row_format, *row))


def test_load_nerfstudio_scene(tmp_path: Path) -> None:
    dataset_root = _write_test_scene(tmp_path)

    scene = load_nerfstudio_scene(dataset_root)

    assert scene.intrinsics.width == 100
    assert scene.intrinsics.height == 100
    assert scene.intrinsics.camera_model == "PINHOLE"
    assert len(scene.frames) == 1
    assert scene.frames[0].file_path == dataset_root / "images" / "frame_00001.png"
    assert np.allclose(scene.frames[0].world_to_camera, np.eye(4))


def test_load_gaussian_cloud(tmp_path: Path) -> None:
    ply_path = tmp_path / "test.ply"
    _write_test_ply(ply_path)

    cloud = load_gaussian_cloud(ply_path)

    assert cloud.vertex_count == 2
    assert np.allclose(cloud.xyz, np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]))
    assert np.allclose(cloud.opacities, np.array([0.1, 0.2], dtype=np.float32))
    assert cloud.scales is not None
    assert cloud.rotations is not None


def test_project_world_points_identity_camera(tmp_path: Path) -> None:
    dataset_root = _write_test_scene(tmp_path)
    scene = load_nerfstudio_scene(dataset_root)
    frame = scene.frames[0]
    intrinsics = scene.intrinsics

    points_world = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    result = project_world_points(points_world, intrinsics, frame)

    assert result.image_points.shape == (3, 2)
    assert np.allclose(result.image_points[0], np.array([50.0, 50.0]), atol=1e-4)
    assert np.allclose(result.image_points[1], np.array([100.0, 50.0]), atol=1e-4)
    assert result.valid_mask.tolist() == [True, False, False]
