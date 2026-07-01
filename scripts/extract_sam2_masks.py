"""Extract per-frame SAM 2 masks into the project manifest format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    save_binary_mask,
    save_frame_masks_manifest,
)
from gsegmenter.segmentation.sam2_extractor import Sam2AutomaticMaskExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SAM 2 masks for a scene.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.is_file())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    extractor = Sam2AutomaticMaskExtractor(
        checkpoint_path=args.checkpoint_path,
        model_config=args.model_config,
    )

    for frame_index, image_path in enumerate(image_paths):
        frame_dir = args.output_root / image_path.stem
        manifest_path = frame_dir / "instances.json"
        if args.skip_existing and manifest_path.exists():
            continue

        predictions = extractor.extract_image(image_path)
        with Image.open(image_path) as image:
            width, height = image.size

        instances: list[MaskInstanceRecord] = []
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

        manifest = FrameMasksManifest(
            frame_index=frame_index,
            image_path=str(image_path),
            image_size=(width, height),
            instances=tuple(instances),
        )
        save_frame_masks_manifest(manifest, manifest_path)
        print(f"Saved {len(instances)} masks for {image_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
