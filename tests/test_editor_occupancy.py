from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import InteriorGSOccupancy, InteriorGSObjectRecord  # noqa: E402
from gsegmenter.editor.occupancy import (  # noqa: E402
    evaluate_interiorgs_object_placement,
    world_xy_to_occupancy_pixels,
)


def _make_occupancy(tmp_path: Path, values: np.ndarray) -> InteriorGSOccupancy:
    image_path = tmp_path / "occupancy.png"
    Image.fromarray(values.astype(np.uint8), mode="L").save(image_path)
    return InteriorGSOccupancy(
        image_path=image_path,
        metadata={
            "scale": 1.0,
            "upper": [4.0, 4.0, 1.0],
            "lower": [0.0, 0.0, 0.0],
        },
    )


def _make_box(origin_xy: tuple[float, float], size_xy: tuple[float, float], z0: float = 0.0, z1: float = 1.0) -> np.ndarray:
    ox, oy = origin_xy
    sx, sy = size_xy
    return np.asarray(
        [
            [ox, oy, z0],
            [ox + sx, oy, z0],
            [ox + sx, oy + sy, z0],
            [ox, oy + sy, z0],
            [ox, oy, z1],
            [ox + sx, oy, z1],
            [ox + sx, oy + sy, z1],
            [ox, oy + sy, z1],
        ],
        dtype=np.float32,
    )


def test_world_xy_to_occupancy_pixels_matches_dataset_formula(tmp_path: Path) -> None:
    occupancy = _make_occupancy(tmp_path, np.full((5, 5), 255, dtype=np.uint8))
    points = np.asarray([[1.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    pixels = world_xy_to_occupancy_pixels(points, occupancy)
    expected = np.asarray([[3.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    assert np.allclose(pixels, expected)


def test_evaluate_interiorgs_object_placement_reports_collision(tmp_path: Path) -> None:
    occupancy = _make_occupancy(
        tmp_path,
        np.asarray(
            [
                [255, 255, 255, 255, 255],
                [255,   0,   0, 255, 255],
                [255,   0,   0, 255, 255],
                [255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )
    record = InteriorGSObjectRecord(
        instance_id=148,
        label="chair",
        bbox_corners=_make_box((2.0, 1.0), (1.0, 1.0)),
        raw_payload={},
    )

    summary = evaluate_interiorgs_object_placement(
        record,
        occupancy,
        object_id=148,
        translation_xyz=np.zeros(3, dtype=np.float32),
        rotation_matrix=np.eye(3, dtype=np.float32),
        max_occupied_fraction=0.05,
        max_unknown_fraction=0.25,
    )

    assert summary.occupied_pixels > 0
    assert summary.valid is False


def test_evaluate_interiorgs_object_placement_accepts_free_space(tmp_path: Path) -> None:
    occupancy = _make_occupancy(tmp_path, np.full((5, 5), 255, dtype=np.uint8))
    record = InteriorGSObjectRecord(
        instance_id=149,
        label="chair",
        bbox_corners=_make_box((1.0, 1.0), (1.0, 1.0)),
        raw_payload={},
    )

    summary = evaluate_interiorgs_object_placement(
        record,
        occupancy,
        object_id=149,
        translation_xyz=np.zeros(3, dtype=np.float32),
        rotation_matrix=np.eye(3, dtype=np.float32),
    )

    assert summary.free_pixels > 0
    assert summary.occupied_pixels == 0
    assert summary.valid is True
