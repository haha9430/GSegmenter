"""GroundingDINO box prompts combined with SAM 2 image masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from gsegmenter.segmentation.sam2_extractor import _mask_bbox


@dataclass(slots=True)
class GroundedMaskPrediction:
    """Furniture-aware mask prediction from a text-conditioned detector box."""

    segmentation: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    area: int
    label: str
    detection_bbox_xyxy: tuple[int, int, int, int]
    detection_score: float


def cxcywh_to_xyxy(
    boxes_cxcywh: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert normalized `(cx, cy, w, h)` boxes into clipped pixel `xyxy` boxes."""

    boxes = np.asarray(boxes_cxcywh, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes shaped `(N, 4)`, got {boxes.shape}")
    xyxy = np.empty_like(boxes, dtype=np.float32)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * float(image_width)
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * float(image_height)
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * float(image_width)
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * float(image_height)
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0.0, float(image_width))
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0.0, float(image_height))
    return xyxy


def filter_boxes_by_area(
    boxes_xyxy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> np.ndarray:
    """Return a mask for boxes whose image-area ratio stays in configured bounds."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes shaped `(N, 4)`, got {boxes.shape}")
    if min_area_ratio < 0.0 or max_area_ratio <= 0.0 or min_area_ratio > max_area_ratio:
        raise ValueError("Area ratio bounds must satisfy 0 <= min <= max.")
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    image_area = max(float(image_width * image_height), 1.0)
    area_ratio = (widths * heights) / image_area
    return (area_ratio >= float(min_area_ratio)) & (area_ratio <= float(max_area_ratio))


class GroundedSam2MaskExtractor:
    """Generate SAM 2 masks from GroundingDINO furniture boxes."""

    def __init__(
        self,
        *,
        detector_backend: str = "transformers",
        grounding_config_path: Path | None = None,
        grounding_checkpoint_path: Path | None = None,
        hf_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_checkpoint_path: Path,
        sam2_model_config: str,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        min_box_area_ratio: float = 0.001,
        max_box_area_ratio: float = 0.45,
    ) -> None:
        if detector_backend not in {"groundingdino", "transformers"}:
            raise ValueError("--detector-backend must be 'groundingdino' or 'transformers'.")
        self.detector_backend = detector_backend
        self.grounding_config_path = Path(grounding_config_path) if grounding_config_path is not None else None
        self.grounding_checkpoint_path = Path(grounding_checkpoint_path) if grounding_checkpoint_path is not None else None
        self.hf_model_id = hf_model_id
        self.sam2_checkpoint_path = Path(sam2_checkpoint_path)
        self.sam2_model_config = sam2_model_config
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self._grounding_model = None
        self._grounding_processor = None
        if self.detector_backend == "groundingdino":
            self._grounding_model = self._build_grounding_model()
        else:
            self._grounding_processor, self._grounding_model = self._build_transformers_detector()
        self._sam2_predictor = self._build_sam2_predictor()

    def _build_grounding_model(self):
        if self.grounding_config_path is None or self.grounding_checkpoint_path is None:
            raise ValueError(
                "The 'groundingdino' backend requires --grounding-config-path and "
                "--grounding-checkpoint-path."
            )
        try:
            from groundingdino.util.inference import load_model
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise ImportError(
                "GroundingDINO is not installed in the current Python environment. "
                "Install GroundingDINO in the SAM2 environment before running grounded extraction."
            ) from exc
        return load_model(
            str(self.grounding_config_path),
            str(self.grounding_checkpoint_path),
            device=self.device,
        )

    def _build_transformers_detector(self):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise ImportError(
                "The transformers detector backend requires `transformers` and `torch`. "
                "Install them in the SAM2 environment before running grounded extraction."
            ) from exc

        processor = AutoProcessor.from_pretrained(self.hf_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.hf_model_id)
        model.to(torch.device(self.device))
        model.eval()
        return processor, model

    def _build_sam2_predictor(self):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise ImportError(
                "SAM 2 image predictor is not installed in the current Python environment."
            ) from exc
        sam2_model = build_sam2(self.sam2_model_config, str(self.sam2_checkpoint_path), device=self.device)
        return SAM2ImagePredictor(sam2_model)

    def _detect_boxes(
        self,
        image_path: Path,
        prompt: str,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        try:
            from groundingdino.util.inference import load_image, predict
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise ImportError("GroundingDINO inference helpers are unavailable.") from exc

        _, grounding_image = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=self._grounding_model,
            image=grounding_image,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        logits_np = logits.detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits)
        boxes_xyxy = cxcywh_to_xyxy(
            boxes_np,
            image_width=image_width,
            image_height=image_height,
        )
        keep = filter_boxes_by_area(
            boxes_xyxy,
            image_width=image_width,
            image_height=image_height,
            min_area_ratio=self.min_box_area_ratio,
            max_area_ratio=self.max_box_area_ratio,
        )
        return boxes_xyxy[keep], logits_np[keep], [str(phrase) for phrase, allowed in zip(phrases, keep) if allowed]

    def _detect_boxes_transformers(
        self,
        image_path: Path,
        prompt: str,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local env.
            raise ImportError("PyTorch is required for the transformers detector backend.") from exc
        if self._grounding_processor is None or self._grounding_model is None:
            raise RuntimeError("Transformers detector backend was not initialized.")

        with Image.open(image_path) as image:
            pil_image = image.convert("RGB")
        inputs = self._grounding_processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        target_sizes = torch.tensor([[image_height, image_width]], device=self.device)
        post_process = self._grounding_processor.post_process_grounded_object_detection
        try:
            results = post_process(
                outputs,
                inputs.get("input_ids"),
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = post_process(
                outputs,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("text_labels", result.get("labels", []))
        boxes_xyxy = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        labels_list = [str(label) for label in labels]

        if boxes_xyxy.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )
        keep = filter_boxes_by_area(
            boxes_xyxy,
            image_width=image_width,
            image_height=image_height,
            min_area_ratio=self.min_box_area_ratio,
            max_area_ratio=self.max_box_area_ratio,
        )
        return boxes_xyxy[keep], scores_np[keep], [label for label, allowed in zip(labels_list, keep) if allowed]

    def extract_image(self, image_path: Path, prompt: str) -> list[GroundedMaskPrediction]:
        """Detect prompted furniture boxes and segment each box with SAM 2."""

        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
            width, height = image.size

        if self.detector_backend == "groundingdino":
            boxes_xyxy, scores, labels = self._detect_boxes(image_path, prompt, width, height)
        else:
            boxes_xyxy, scores, labels = self._detect_boxes_transformers(image_path, prompt, width, height)
        if boxes_xyxy.shape[0] == 0:
            return []

        self._sam2_predictor.set_image(rgb)
        predictions: list[GroundedMaskPrediction] = []
        for box, detection_score, label in zip(boxes_xyxy, scores, labels, strict=True):
            masks, mask_scores, _ = self._sam2_predictor.predict(
                box=box.astype(np.float32),
                multimask_output=False,
            )
            mask = np.asarray(masks[0], dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            mask_score = float(np.asarray(mask_scores).reshape(-1)[0]) if mask_scores is not None else 1.0
            predictions.append(
                GroundedMaskPrediction(
                    segmentation=mask,
                    bbox_xyxy=_mask_bbox(mask),
                    score=float(mask_score * float(detection_score)),
                    area=int(mask.sum()),
                    label=str(label),
                    detection_bbox_xyxy=tuple(int(round(value)) for value in box.tolist()),
                    detection_score=float(detection_score),
                )
            )

        predictions.sort(key=lambda item: item.area, reverse=True)
        return predictions
