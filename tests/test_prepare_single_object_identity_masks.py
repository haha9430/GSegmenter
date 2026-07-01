from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gsegmenter.segmentation.mask_io import load_frame_masks_manifest, save_binary_mask
from scripts.prepare_single_object_identity_masks import main as prepare_single_object_main


def test_prepare_single_object_identity_masks_unifies_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    for frame_index, frame_name in enumerate(("DSCF0001", "DSCF0002")):
        frame_dir = source_root / frame_name
        frame_dir.mkdir(parents=True)
        tv_mask = np.zeros((10, 10), dtype=bool)
        tv_mask[2:6, 2:6] = True
        other_mask = np.zeros((10, 10), dtype=bool)
        other_mask[6:9, 6:9] = True
        save_binary_mask(tv_mask, frame_dir / "mask_tv.png")
        save_binary_mask(other_mask, frame_dir / "mask_other.png")
        payload = {
            "frame_index": frame_index,
            "image_path": f"C:/scene/images_2/{frame_name}.JPG",
            "image_size": [10, 10],
            "instances": [
                {
                    "instance_id": 3 + frame_index,
                    "bbox_xyxy": [2, 2, 5, 5],
                    "score": 0.8,
                    "area": 16,
                    "mask_path": "mask_tv.png",
                    "label": "television tv",
                },
                {
                    "instance_id": 9,
                    "bbox_xyxy": [6, 6, 8, 8],
                    "score": 0.9,
                    "area": 9,
                    "mask_path": "mask_other.png",
                    "label": "chair",
                },
            ],
        }
        (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "tv_only"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_single_object_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--label-regex",
            "television|tv",
            "--object-name",
            "tv_01",
            "--object-id",
            "0",
            "--min-score",
            "0.5",
        ],
    )

    assert prepare_single_object_main() == 0

    manifest_a = load_frame_masks_manifest(output_root / "frame_00000" / "instances.json")
    manifest_b = load_frame_masks_manifest(output_root / "frame_00001" / "instances.json")
    summary = json.loads((output_root / "identity_mask_manifest.json").read_text(encoding="utf-8"))

    assert summary["class_count"] == 1
    assert summary["total_instances"] == 2
    assert manifest_a.instances[0].instance_id == 0
    assert manifest_b.instances[0].instance_id == 0
    assert manifest_a.instances[0].mask_path == "mask_0000.png"


def test_prepare_single_object_identity_masks_adds_background_class(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    frame_dir = source_root / "DSCF0001"
    frame_dir.mkdir(parents=True)
    tv_mask = np.zeros((6, 8), dtype=bool)
    tv_mask[1:5, 2:6] = True
    save_binary_mask(tv_mask, frame_dir / "mask_tv.png")
    payload = {
        "frame_index": 0,
        "image_path": "C:/scene/images_2/DSCF0001.JPG",
        "image_size": [8, 6],
        "instances": [
            {
                "instance_id": 7,
                "bbox_xyxy": [2, 1, 5, 4],
                "score": 0.9,
                "area": 16,
                "mask_path": "mask_tv.png",
                "label": "television tv",
            },
        ],
    }
    (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "tv_only_bg"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_single_object_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--label-regex",
            "television|tv",
            "--object-name",
            "tv_01",
            "--object-id",
            "1",
            "--add-background-class",
            "--background-id",
            "0",
            "--min-score",
            "0.5",
        ],
    )

    assert prepare_single_object_main() == 0

    manifest = load_frame_masks_manifest(output_root / "frame_00000" / "instances.json")
    summary = json.loads((output_root / "identity_mask_manifest.json").read_text(encoding="utf-8"))

    assert summary["class_count"] == 2
    assert summary["classes"] == [
        {"raw_key": "background", "global_id": 0},
        {"raw_key": "tv_01", "global_id": 1},
    ]
    assert [instance.instance_id for instance in manifest.instances] == [0, 1]
    assert manifest.instances[0].mask_path == "mask_background.png"
    assert manifest.instances[0].area == 48


def test_prepare_single_object_identity_masks_filters_source_frame_and_bbox_center(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    for frame_name, bbox in (("DSCF1000", [0, 1, 3, 4]), ("DSCF2000", [5, 1, 7, 4])):
        frame_dir = source_root / frame_name
        frame_dir.mkdir(parents=True)
        mask = np.zeros((6, 8), dtype=bool)
        x0, y0, x1, y1 = bbox
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        save_binary_mask(mask, frame_dir / "mask_sofa.png")
        payload = {
            "frame_index": 0,
            "image_path": f"C:/scene/images_2/{frame_name}.JPG",
            "image_size": [8, 6],
            "instances": [
                {
                    "instance_id": 7,
                    "bbox_xyxy": bbox,
                    "score": 0.9,
                    "area": int(mask.sum()),
                    "mask_path": "mask_sofa.png",
                    "label": "sofa couch",
                },
            ],
        }
        (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "sofa_left"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_single_object_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--label-regex",
            "sofa|couch",
            "--object-name",
            "sofa_01",
            "--object-id",
            "1",
            "--source-frame-regex",
            "DSCF1000|DSCF2000",
            "--bbox-center-x-max",
            "0.5",
            "--min-score",
            "0.5",
        ],
    )

    assert prepare_single_object_main() == 0

    summary = json.loads((output_root / "identity_mask_manifest.json").read_text(encoding="utf-8"))

    assert summary["frame_count"] == 1
    assert summary["frames"][0]["source_frame"] == "DSCF1000"
    assert summary["frames"][0]["selected"][0]["bbox_center"][0] < 0.5
