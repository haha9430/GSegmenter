from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.data.nerfstudio_scene import load_nerfstudio_scene
from gsegmenter.mapping.lifting import build_frame_vote_evidence, build_front_surface_mask, collect_mask_hits
from gsegmenter.mapping.lifting import build_depth_consistency_mask
from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    load_binary_mask,
    load_frame_masks_manifest,
    save_binary_mask,
    save_frame_masks_manifest,
)


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
    (dataset_root / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    return dataset_root


def test_mask_manifest_roundtrip(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True
    frame_dir = tmp_path / "frame_00001"
    mask_path = frame_dir / "mask_0000.png"
    save_binary_mask(mask, mask_path)

    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="dummy.png",
        image_size=(8, 8),
        instances=(
            MaskInstanceRecord(
                instance_id=0,
                bbox_xyxy=(3, 2, 6, 5),
                score=0.9,
                area=9,
                mask_path="mask_0000.png",
            ),
        ),
    )
    manifest_path = frame_dir / "instances.json"
    save_frame_masks_manifest(manifest, manifest_path)

    loaded_manifest = load_frame_masks_manifest(manifest_path)
    loaded_mask = load_binary_mask(mask_path)

    assert loaded_manifest.instances[0].bbox_xyxy == (3, 2, 6, 5)
    assert loaded_mask.sum() == 9


def test_collect_mask_hits() -> None:
    image_points = np.array([[1.1, 1.2], [2.9, 2.1], [0.5, 0.5]], dtype=np.float32)
    valid_mask = np.array([True, True, False])
    binary_mask = np.zeros((4, 4), dtype=bool)
    binary_mask[1, 1] = True
    binary_mask[2, 2] = True

    hits = collect_mask_hits(image_points, valid_mask, binary_mask)

    assert hits.tolist() == [0, 1]


def test_collect_mask_hits_discards_rounded_out_of_bounds_points() -> None:
    image_points = np.array([[3.0, 1.0], [1.2, 1.2]], dtype=np.float32)
    valid_mask = np.array([True, True])
    binary_mask = np.zeros((3, 3), dtype=bool)
    binary_mask[1, 1] = True

    hits = collect_mask_hits(image_points, valid_mask, binary_mask)

    assert hits.tolist() == [1]


def test_build_frame_vote_evidence(tmp_path: Path) -> None:
    dataset_root = _write_test_scene(tmp_path)
    scene = load_nerfstudio_scene(dataset_root)
    frame = scene.frames[0]

    frame_dir = tmp_path / "masks" / "frame_00001"
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:56, 45:56] = True
    save_binary_mask(mask, frame_dir / "mask_0000.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path=str(frame.file_path),
        image_size=(100, 100),
        instances=(
            MaskInstanceRecord(
                instance_id=0,
                bbox_xyxy=(45, 45, 56, 56),
                score=0.75,
                area=int(mask.sum()),
                mask_path="mask_0000.png",
            ),
        ),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")

    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    evidences = build_frame_vote_evidence(
        gaussian_xyz=gaussian_xyz,
        intrinsics=scene.intrinsics,
        frame=frame,
        manifest=manifest,
        frame_dir=frame_dir,
    )

    assert len(evidences) == 1
    assert evidences[0].gaussian_indices.tolist() == [0]
    assert np.allclose(evidences[0].weights, np.array([0.75], dtype=np.float32))


def test_build_frame_vote_evidence_applies_quality_and_valid_mask(tmp_path: Path) -> None:
    dataset_root = _write_test_scene(tmp_path)
    scene = load_nerfstudio_scene(dataset_root)
    frame = scene.frames[0]

    frame_dir = tmp_path / "masks" / "frame_00001"
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:56, 45:56] = True
    save_binary_mask(mask, frame_dir / "mask_0000.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path=str(frame.file_path),
        image_size=(100, 100),
        instances=(
            MaskInstanceRecord(
                instance_id=0,
                bbox_xyxy=(45, 45, 56, 56),
                score=0.8,
                area=int(mask.sum()),
                mask_path="mask_0000.png",
            ),
        ),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")

    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.02, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    evidences = build_frame_vote_evidence(
        gaussian_xyz=gaussian_xyz,
        intrinsics=scene.intrinsics,
        frame=frame,
        manifest=manifest,
        frame_dir=frame_dir,
        quality_weights=np.array([0.5, 1.0], dtype=np.float32),
        gaussian_valid_mask=np.array([True, False]),
    )

    assert len(evidences) == 1
    assert evidences[0].gaussian_indices.tolist() == [0]
    assert np.allclose(evidences[0].weights, np.array([0.4], dtype=np.float32))


def test_build_frame_vote_evidence_can_keep_front_surface_only(tmp_path: Path) -> None:
    dataset_root = _write_test_scene(tmp_path)
    scene = load_nerfstudio_scene(dataset_root)
    frame = scene.frames[0]

    frame_dir = tmp_path / "masks" / "frame_00001"
    mask = np.zeros((100, 100), dtype=bool)
    mask[48:53, 48:53] = True
    save_binary_mask(mask, frame_dir / "mask_0000.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path=str(frame.file_path),
        image_size=(100, 100),
        instances=(
            MaskInstanceRecord(
                instance_id=0,
                bbox_xyxy=(48, 48, 53, 53),
                score=0.8,
                area=int(mask.sum()),
                mask_path="mask_0000.png",
            ),
        ),
    )

    gaussian_xyz = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )

    evidences = build_frame_vote_evidence(
        gaussian_xyz=gaussian_xyz,
        intrinsics=scene.intrinsics,
        frame=frame,
        manifest=manifest,
        frame_dir=frame_dir,
        front_surface_only=True,
        front_surface_depth_margin=0.1,
    )

    assert len(evidences) == 1
    assert evidences[0].gaussian_indices.tolist() == [0]


def test_build_depth_consistency_mask_rejects_points_behind_surface() -> None:
    image_points = np.array(
        [
            [1.2, 1.2],
            [1.4, 1.4],
            [2.2, 1.2],
            [2.4, 1.4],
            [3.2, 1.2],
            [3.4, 1.4],
        ],
        dtype=np.float32,
    )
    depths = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0], dtype=np.float32)
    valid_mask = np.ones((6,), dtype=bool)
    depth_map = np.ones((5, 5), dtype=np.float32)

    keep = build_depth_consistency_mask(
        image_points,
        depths,
        valid_mask,
        depth_map,
        fit_max_points=100,
        behind_margin_ratio=0.1,
        behind_min_margin=0.05,
    )

    assert keep.tolist() == [True, False, True, False, True, False]


def test_build_front_surface_mask_keeps_nearest_depth_per_pixel() -> None:
    image_points = np.array(
        [
            [1.2, 1.2],
            [1.4, 1.4],
            [2.2, 1.2],
            [2.4, 1.4],
        ],
        dtype=np.float32,
    )
    depths = np.array([1.0, 1.02, 2.0, 2.2], dtype=np.float32)
    valid_mask = np.ones((4,), dtype=bool)

    keep = build_front_surface_mask(
        image_points,
        depths,
        valid_mask,
        (4, 4),
        depth_margin=0.05,
    )

    assert keep.tolist() == [True, True, True, False]
