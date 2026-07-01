"""Extract furniture-aware masks with GroundingDINO boxes and SAM 2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.grounded_sam2 import GroundedSam2MaskExtractor
from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    save_binary_mask,
)


DEFAULT_FURNITURE_PROMPT = (
    "chair . table . sofa . couch . cabinet . shelf . bookshelf . desk . "
    "bed . television . tv . wardrobe . drawer . lamp . stool . ottoman . "
    "nightstand . dresser"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract furniture masks by detecting prompted boxes with GroundingDINO and segmenting them with SAM 2."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("transformers", "groundingdino"), default="transformers")
    parser.add_argument("--hf-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-config-path", type=Path, default=None)
    parser.add_argument("--grounding-checkpoint-path", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint-path", type=Path, required=True)
    parser.add_argument("--sam2-model-config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_FURNITURE_PROMPT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _write_manifest_with_labels(
    manifest: FrameMasksManifest,
    output_path: Path,
    instance_extras: list[dict[str, object]],
) -> None:
    payload = {
        "frame_index": manifest.frame_index,
        "image_path": manifest.image_path,
        "image_size": list(manifest.image_size),
        "instances": [],
    }
    for instance, extras in zip(manifest.instances, instance_extras, strict=True):
        record = asdict(instance)
        record.update(extras)
        payload["instances"].append(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.is_file())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    extractor = GroundedSam2MaskExtractor(
        detector_backend=args.detector_backend,
        grounding_config_path=args.grounding_config_path,
        grounding_checkpoint_path=args.grounding_checkpoint_path,
        hf_model_id=args.hf_model_id,
        sam2_checkpoint_path=args.sam2_checkpoint_path,
        sam2_model_config=args.sam2_model_config,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        min_box_area_ratio=args.min_box_area_ratio,
        max_box_area_ratio=args.max_box_area_ratio,
    )

    for frame_index, image_path in enumerate(image_paths):
        frame_dir = args.output_root / image_path.stem
        manifest_path = frame_dir / "instances.json"
        if args.skip_existing and manifest_path.exists():
            continue

        predictions = extractor.extract_image(image_path, args.prompt)
        with Image.open(image_path) as image:
            width, height = image.size

        instances: list[MaskInstanceRecord] = []
        instance_extras: list[dict[str, object]] = []
        for instance_id, prediction in enumerate(predictions):
            mask_filename = f"mask_{instance_id:04d}.png"
            save_binary_mask(prediction.segmentation, frame_dir / mask_filename)
            instances.append(
                MaskInstanceRecord(
                    instance_id=instance_id,
                    bbox_xyxy=prediction.bbox_xyxy,
                    score=prediction.score,
                    area=prediction.area,
                    mask_path=mask_filename,
                )
            )
            instance_extras.append(
                {
                    "label": prediction.label,
                    "detection_bbox_xyxy": list(prediction.detection_bbox_xyxy),
                    "detection_score": prediction.detection_score,
                }
            )

        manifest = FrameMasksManifest(
            frame_index=frame_index,
            image_path=str(image_path),
            image_size=(width, height),
            instances=tuple(instances),
        )
        _write_manifest_with_labels(manifest, manifest_path, instance_extras)
        print(f"Saved {len(instances)} grounded masks for {image_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
