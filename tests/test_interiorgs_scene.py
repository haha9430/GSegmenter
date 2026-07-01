from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.data import load_interiorgs_scene


def test_load_interiorgs_scene_parses_expected_files(tmp_path: Path) -> None:
    scene_root = tmp_path / "0001_839920"
    scene_root.mkdir(parents=True)

    (scene_root / "3dgs_compressed.ply").write_bytes(b"ply\n")
    (scene_root / "occupancy.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (scene_root / "labels.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "instance_id": 3,
                        "label": "Chair",
                        "bbox": [[0, 0, 0]] * 8,
                    },
                    {
                        "instance_id": 4,
                        "label": "Table",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (scene_root / "occupancy.json").write_text(
        json.dumps({"origin": [0, 0], "resolution": 0.05}),
        encoding="utf-8",
    )
    (scene_root / "structure.json").write_text(
        json.dumps({"rooms": [{"room_type": "Living Room"}], "walls": [{}, {}]}),
        encoding="utf-8",
    )

    scene = load_interiorgs_scene(scene_root)

    assert scene.gaussian_ply_path.name == "3dgs_compressed.ply"
    assert len(scene.objects) == 2
    assert scene.objects[0].instance_id == 3
    assert scene.objects[0].label == "Chair"
    assert scene.objects[0].bbox_corners is not None
    assert scene.structure.room_count == 1
    assert scene.structure.wall_count == 2


def test_load_interiorgs_scene_accepts_list_style_labels_and_dict_bbox(tmp_path: Path) -> None:
    scene_root = tmp_path / "0003_839989"
    scene_root.mkdir(parents=True)

    (scene_root / "3dgs_compressed.ply").write_bytes(b"ply\n")
    (scene_root / "occupancy.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (scene_root / "labels.json").write_text(
        json.dumps(
            [
                {
                    "ins_id": "48",
                    "label": "door",
                    "bounding_box": [{"x": 0, "y": 0, "z": 0}] * 8,
                },
                {
                    "ins_id": "",
                    "label": "room",
                },
            ]
        ),
        encoding="utf-8",
    )
    (scene_root / "occupancy.json").write_text(
        json.dumps({"scale": 0.05, "center": [0, 0, 0]}),
        encoding="utf-8",
    )
    (scene_root / "structure.json").write_text(
        json.dumps({"rooms": [{"profile": []}], "walls": []}),
        encoding="utf-8",
    )

    scene = load_interiorgs_scene(scene_root)

    assert len(scene.objects) == 2
    assert scene.objects[0].instance_id == 48
    assert scene.objects[0].bbox_corners is not None
    assert scene.objects[1].instance_id is None
    assert scene.objects[1].label == "room"


def test_load_interiorgs_scene_requires_all_expected_files(tmp_path: Path) -> None:
    scene_root = tmp_path / "0002_839955"
    scene_root.mkdir(parents=True)
    (scene_root / "3dgs_compressed.ply").write_bytes(b"ply\n")

    try:
        load_interiorgs_scene(scene_root)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected load_interiorgs_scene to fail on incomplete scene folder.")

    assert "labels.json" in message
    assert "occupancy.json" in message
    assert "structure.json" in message
