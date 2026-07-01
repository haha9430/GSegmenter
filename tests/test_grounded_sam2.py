from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.grounded_sam2 import (  # noqa: E402
    cxcywh_to_xyxy,
    filter_boxes_by_area,
)


def test_cxcywh_to_xyxy_clips_normalized_boxes() -> None:
    boxes = np.array(
        [
            [0.5, 0.5, 0.25, 0.5],
            [0.0, 1.0, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    converted = cxcywh_to_xyxy(boxes, image_width=200, image_height=100)

    assert np.allclose(converted[0], np.array([75.0, 25.0, 125.0, 75.0], dtype=np.float32))
    assert np.allclose(converted[1], np.array([0.0, 75.0, 50.0, 100.0], dtype=np.float32))


def test_filter_boxes_by_area_keeps_configured_area_range() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 90.0, 90.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    keep = filter_boxes_by_area(
        boxes,
        image_width=100,
        image_height=100,
        min_area_ratio=0.005,
        max_area_ratio=0.5,
    )

    assert keep.tolist() == [True, False, False]
