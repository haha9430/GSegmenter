"""Extract per-image monocular depth maps with a Transformers depth model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract monocular depth maps for an image directory.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _normalize_to_uint16(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros(depth.shape, dtype=np.uint16)
    lo = float(np.percentile(depth[finite], 1.0))
    hi = float(np.percentile(depth[finite], 99.0))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 65535.0).astype(np.uint16)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as torch_functional
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    image_paths = sorted(path for path in args.images_dir.iterdir() if path.is_file())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    summaries: list[dict[str, object]] = []
    for image_path in image_paths:
        frame_dir = args.output_root / image_path.stem
        depth_path = frame_dir / "depth.npy"
        if args.skip_existing and depth_path.exists():
            continue

        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            inputs = processor(images=rgb, return_tensors="pt")

        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.predicted_depth.unsqueeze(1)
            resized = torch_functional.interpolate(
                prediction,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = resized.detach().cpu().numpy().astype(np.float32)

        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, depth)
        Image.fromarray(_normalize_to_uint16(depth), mode="I;16").save(frame_dir / "depth_preview.png")
        summary = {
            "image_path": str(image_path),
            "image_size": [width, height],
            "depth_path": "depth.npy",
            "preview_path": "depth_preview.png",
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
            "depth_median": float(np.nanmedian(depth)),
        }
        (frame_dir / "depth_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(f"Saved monocular depth for {image_path.name}")

    (args.output_root / "depth_summary.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "image_count": len(image_paths),
                "processed_count": len(summaries),
                "frames": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
