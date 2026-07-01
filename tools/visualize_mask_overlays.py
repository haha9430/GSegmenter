"""Render 2D mask manifests as image overlays for quick detector QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.mask_io import load_binary_mask


PALETTE = np.asarray(
    [
        [255, 48, 48],
        [48, 190, 80],
        [64, 120, 255],
        [255, 180, 36],
        [190, 70, 230],
        [36, 200, 200],
        [255, 110, 170],
        [150, 220, 48],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create image overlays from mask instances.json files.")
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-stems", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--draw-detection-box", action="store_true")
    return parser.parse_args()


def _manifest_dirs(masks_root: Path, frame_stems: list[str] | None, limit: int | None) -> list[Path]:
    if frame_stems:
        dirs = [masks_root / stem for stem in frame_stems]
    else:
        dirs = [path for path in sorted(masks_root.iterdir()) if (path / "instances.json").exists()]
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    bg = (max(color[0] - 40, 0), max(color[1] - 40, 0), max(color[2] - 40, 0))
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=bg,
    )
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def _render_overlay(frame_dir: Path, output_path: Path, alpha: float, draw_detection_box: bool) -> None:
    manifest = json.loads((frame_dir / "instances.json").read_text(encoding="utf-8"))
    image_path = Path(manifest["image_path"])
    with Image.open(image_path) as image:
        base = np.asarray(image.convert("RGB"), dtype=np.float32)

    overlay = base.copy()
    draw_image = Image.fromarray(base.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(draw_image)

    for index, instance in enumerate(manifest["instances"]):
        color = PALETTE[index % len(PALETTE)]
        mask = load_binary_mask(frame_dir / instance["mask_path"])
        if mask.shape[:2] != base.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {base.shape[:2]} for {frame_dir}"
            )
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color.astype(np.float32)
        x0, y0, x1, y1 = (int(value) for value in instance["bbox_xyxy"])
        draw.rectangle((x0, y0, x1, y1), outline=tuple(int(v) for v in color), width=3)
        if draw_detection_box and "detection_bbox_xyxy" in instance:
            dx0, dy0, dx1, dy1 = (int(value) for value in instance["detection_bbox_xyxy"])
            draw.rectangle((dx0, dy0, dx1, dy1), outline=tuple(int(v) for v in color), width=1)
        label = str(instance.get("label", f"id={instance['instance_id']}"))
        score = float(instance.get("detection_score", instance.get("score", 0.0)))
        _draw_label(draw, (x0 + 4, max(y0 + 4, 0)), f"{index}: {label} {score:.2f}", tuple(int(v) for v in color))

    # Draw boxes/labels over blended masks.
    label_layer = np.asarray(draw_image.convert("RGB"), dtype=np.float32)
    non_mask = np.all(label_layer == base, axis=2)
    output = overlay.astype(np.uint8)
    output[~non_mask] = label_layer[~non_mask].astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(output_path)


def main() -> int:
    args = parse_args()
    frame_stems = list(args.frame_stems) if args.frame_stems else None
    frame_dirs = _manifest_dirs(args.masks_root, frame_stems, args.limit)
    if not frame_dirs:
        raise ValueError(f"No mask manifests found under {args.masks_root}")
    for frame_dir in frame_dirs:
        output_path = args.output_root / f"{frame_dir.name}_overlay.png"
        _render_overlay(frame_dir, output_path, float(args.alpha), bool(args.draw_detection_box))
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
