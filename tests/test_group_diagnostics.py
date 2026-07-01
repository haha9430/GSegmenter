from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.data.nerfstudio_scene import load_nerfstudio_scene
from gsegmenter.mapping.group_diagnostics import find_best_frame_for_group, select_group_ids


def _write_scene(root: Path) -> Path:
    dataset_root = root / "scene"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame_00001.png").write_bytes(b"")
    (images_dir / "frame_00002.png").write_bytes(b"")

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
            },
            {
                "file_path": "images/frame_00002.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, -0.5],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        ],
    }
    (dataset_root / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    return dataset_root


def test_select_group_ids_filters() -> None:
    groups = [
        {"global_object_id": 10, "gaussian_count": 100},
        {"global_object_id": 11, "gaussian_count": 50},
        {"global_object_id": 12, "gaussian_count": 25},
    ]
    selected = select_group_ids(
        groups,
        top_k=2,
        skip_largest_n=1,
        min_group_size=20,
        exclude_object_ids=[12],
    )
    assert selected == [11]


def test_find_best_frame_for_group(tmp_path: Path) -> None:
    dataset_root = _write_scene(tmp_path)
    scene = load_nerfstudio_scene(dataset_root)
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [0.4, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    object_ids = np.array([5, 5, 7], dtype=np.int64)

    summary = find_best_frame_for_group(scene, points, object_ids, 5)

    assert summary.object_id == 5
    assert summary.gaussian_count == 2
    assert summary.best_visible_count == 2
    assert summary.best_frame_index == 0
    assert np.isclose(summary.visibility_ratio, 1.0)
