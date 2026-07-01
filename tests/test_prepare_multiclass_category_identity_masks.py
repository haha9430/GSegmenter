from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gsegmenter.segmentation.mask_io import load_frame_masks_manifest, save_binary_mask
from scripts.prepare_multiclass_category_identity_masks import main as prepare_multiclass_main


def test_prepare_multiclass_category_identity_masks_adds_background_and_categories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    frame_dir = source_root / "DSCF0001"
    frame_dir.mkdir(parents=True)
    tv_mask = np.zeros((8, 8), dtype=bool)
    tv_mask[1:4, 1:4] = True
    chair_mask = np.zeros((8, 8), dtype=bool)
    chair_mask[4:7, 4:7] = True
    save_binary_mask(tv_mask, frame_dir / "mask_tv.png")
    save_binary_mask(chair_mask, frame_dir / "mask_chair.png")
    payload = {
        "frame_index": 0,
        "image_path": "C:/scene/images_2/DSCF0001.JPG",
        "image_size": [8, 8],
        "instances": [
            {
                "instance_id": 3,
                "bbox_xyxy": [1, 1, 3, 3],
                "score": 0.8,
                "detection_score": 0.7,
                "area": int(tv_mask.sum()),
                "mask_path": "mask_tv.png",
                "label": "television tv",
            },
            {
                "instance_id": 4,
                "bbox_xyxy": [4, 4, 6, 6],
                "score": 0.9,
                "detection_score": 0.6,
                "area": int(chair_mask.sum()),
                "mask_path": "mask_chair.png",
                "label": "chair",
            },
        ],
    }
    (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "multi"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_multiclass_category_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--category",
            "tv",
            "1",
            "television|tv",
            "--category",
            "chair",
            "2",
            "chair",
            "--min-score",
            "0.5",
        ],
    )

    assert prepare_multiclass_main() == 0

    manifest = load_frame_masks_manifest(output_root / "frame_00000" / "instances.json")
    summary = json.loads((output_root / "identity_mask_manifest.json").read_text(encoding="utf-8"))

    assert [instance.instance_id for instance in manifest.instances] == [0, 1, 2]
    assert manifest.instances[0].mask_path == "mask_background.png"
    assert summary["class_count"] == 3
    assert summary["category_counts"] == {"tv": 1, "chair": 1}


def test_prepare_multiclass_category_identity_masks_can_limit_per_category(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "grounded"
    frame_dir = source_root / "DSCF0001"
    frame_dir.mkdir(parents=True)
    for index, score in enumerate((0.9, 0.7)):
        mask = np.zeros((6, 6), dtype=bool)
        mask[index : index + 2, index : index + 2] = True
        save_binary_mask(mask, frame_dir / f"mask_{index}.png")
    payload = {
        "frame_index": 0,
        "image_path": "C:/scene/images_2/DSCF0001.JPG",
        "image_size": [6, 6],
        "instances": [
            {
                "instance_id": index,
                "bbox_xyxy": [index, index, index + 1, index + 1],
                "score": score,
                "detection_score": score,
                "area": 4,
                "mask_path": f"mask_{index}.png",
                "label": "chair",
            }
            for index, score in enumerate((0.7, 0.9))
        ],
    }
    (frame_dir / "instances.json").write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "limited"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_multiclass_category_identity_masks.py",
            "--masks-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--category",
            "chair",
            "2",
            "chair",
            "--max-instances-per-category-per-frame",
            "1",
        ],
    )

    assert prepare_multiclass_main() == 0

    manifest = load_frame_masks_manifest(output_root / "frame_00000" / "instances.json")

    assert len(manifest.instances) == 2
    assert manifest.instances[1].bbox_xyxy == (1, 1, 2, 2)
