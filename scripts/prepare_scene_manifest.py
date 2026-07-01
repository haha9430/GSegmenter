"""Build a reusable scene manifest from NerfStudio data and exported Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_nerfstudio_scene
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a scene manifest for grouping.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _resolve_property_summary(properties: dict[str, np.ndarray]) -> dict[str, object]:
    """Summarize numeric PLY channels without copying large arrays."""

    summary: dict[str, object] = {}
    for name, values in properties.items():
        summary[name] = {
            "dtype": str(values.dtype),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main() -> int:
    args = parse_args()
    scene = load_nerfstudio_scene(args.dataset_root)
    cloud = load_gaussian_cloud(args.ply_path)

    xyz = cloud.xyz
    manifest = {
        "dataset_root": str(scene.dataset_root),
        "transforms_path": str(scene.transforms_path),
        "image_count": len(scene.frames),
        "image_paths": [str(frame.file_path) for frame in scene.frames],
        "camera_model": scene.intrinsics.camera_model,
        "image_width": scene.intrinsics.width,
        "image_height": scene.intrinsics.height,
        "fl_x": scene.intrinsics.fl_x,
        "fl_y": scene.intrinsics.fl_y,
        "cx": scene.intrinsics.cx,
        "cy": scene.intrinsics.cy,
        "distortion_params": list(scene.intrinsics.distortion_params),
        "coordinate_convention": {
            "camera_to_world": "Transforms camera-frame coordinates into world space.",
            "world_to_camera": "Inverse of camera_to_world, used for projection.",
        },
        "gaussian_count": cloud.vertex_count,
        "gaussian_property_names": sorted(cloud.properties.keys()),
        "gaussian_property_summary": _resolve_property_summary(cloud.properties),
        "gaussian_bounds": {
            "min_xyz": xyz.min(axis=0).tolist(),
            "max_xyz": xyz.max(axis=0).tolist(),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote scene manifest to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
