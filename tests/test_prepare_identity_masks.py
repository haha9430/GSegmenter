from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gsegmenter.segmentation.mask_io import load_frame_masks_manifest, save_binary_mask
from scripts.prepare_identity_masks import main as prepare_identity_masks_main


def test_prepare_identity_masks_converts_labels_to_scene_global_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    for frame_name, label in (("DSCF0001", "sofa couch"), ("DSCF0002", "chair")):
        frame_dir = source_root / frame_name
        frame_dir.mkdir(parents=True)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        save_binary_mask(mask, frame_dir / "mask_0000.png")
        payload = {
            "frame_index": 0 if frame_name.endswith("1") else 1,
            "image_path": f"C:/scene/images_2/{frame_name}.JPG",
            "image_size": [4, 4],
            "instances": [
                {
                    "instance_id": 0,
                    "bbox_xyxy": [1, 1, 2, 2],
                    "score": 0.9,
                    "area": 4,
                    "mask_path": "mask_0000.png",
                    "label": label,
                }
            ],
        }
        (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "identity"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--id-mode",
            "label-family",
        ],
    )

    assert prepare_identity_masks_main() == 0

    manifest_a = load_frame_masks_manifest(output_root / "frame_00000" / "instances.json")
    manifest_b = load_frame_masks_manifest(output_root / "frame_00001" / "instances.json")
    summary = json.loads((output_root / "identity_mask_manifest.json").read_text(encoding="utf-8"))

    assert summary["class_count"] == 1
    assert summary["classes"][0]["raw_key"] == "seat"
    assert manifest_a.instances[0].instance_id == 0
    assert manifest_b.instances[0].instance_id == 0


def test_prepare_identity_masks_drops_unmapped_label_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    frame_dir = source_root / "DSCF0001"
    frame_dir.mkdir(parents=True)
    mask = np.ones((3, 3), dtype=bool)
    save_binary_mask(mask, frame_dir / "mask_0000.png")
    payload = {
        "frame_index": 0,
        "image_path": "C:/scene/images_2/DSCF0001.JPG",
        "image_size": [3, 3],
        "instances": [
            {
                "instance_id": 0,
                "bbox_xyxy": [0, 0, 2, 2],
                "score": 0.9,
                "area": 9,
                "mask_path": "mask_0000.png",
                "label": "unmapped thing",
            }
        ],
    }
    (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "identity"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--id-mode",
            "label-family",
        ],
    )

    try:
        prepare_identity_masks_main()
    except ValueError as error:
        assert "No identity classes" in str(error)
    else:
        raise AssertionError("Expected unmapped-only masks to fail fast.")
