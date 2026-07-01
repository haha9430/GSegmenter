"""Thin SAM 2 integration for automatic mask extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class AutomaticMaskPrediction:
    """Normalized SAM-style automatic mask prediction."""

    segmentation: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    area: int


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute an inclusive-exclusive bounding box from a boolean mask."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute bounding box for an empty mask.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class Sam2AutomaticMaskExtractor:
    """Automatic mask extractor backed by SAM 2 when available."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        model_config: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config = model_config
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._mask_generator = self._build_generator()

    def _build_generator(self):
        """Instantiate the optional SAM 2 automatic mask generator."""

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise ImportError(
                "SAM 2 is not installed in the current Python environment. "
                "Install the official SAM 2 package before running mask extraction."
            ) from exc

        predictor = build_sam2(self.model_config, str(self.checkpoint_path))
        return SAM2AutomaticMaskGenerator(
            predictor,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )

    def extract_image(self, image_path: Path) -> list[AutomaticMaskPrediction]:
        """Run SAM 2 automatic mask generation on a single RGB image."""

        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))

        raw_masks = self._mask_generator.generate(rgb)
        predictions: list[AutomaticMaskPrediction] = []
        for raw_mask in raw_masks:
            segmentation = np.asarray(raw_mask["segmentation"], dtype=bool)
            if segmentation.ndim != 2 or not segmentation.any():
                continue

            bbox = raw_mask.get("bbox")
            if bbox is None:
                bbox_xyxy = _mask_bbox(segmentation)
            else:
                x, y, width, height = bbox
                bbox_xyxy = (
                    int(round(x)),
                    int(round(y)),
                    int(round(x + width)),
                    int(round(y + height)),
                )

            score = float(raw_mask.get("predicted_iou", raw_mask.get("stability_score", 1.0)))
            predictions.append(
                AutomaticMaskPrediction(
                    segmentation=segmentation,
                    bbox_xyxy=bbox_xyxy,
                    score=score,
                    area=int(segmentation.sum()),
                )
            )

        predictions.sort(key=lambda item: item.area, reverse=True)
        return predictions
