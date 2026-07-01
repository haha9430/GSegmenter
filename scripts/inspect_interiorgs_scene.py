"""Inspect a downloaded InteriorGS scene folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import load_interiorgs_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an InteriorGS scene folder.")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--top-k-labels", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = load_interiorgs_scene(args.scene_root)

    label_hist: dict[str, int] = {}
    with_bbox = 0
    for record in scene.objects:
        label_hist[record.label] = label_hist.get(record.label, 0) + 1
        if record.bbox_corners is not None:
            with_bbox += 1

    top_labels = sorted(label_hist.items(), key=lambda item: (-item[1], item[0]))[: args.top_k_labels]
    payload = {
        "scene_root": str(scene.scene_root),
        "gaussian_ply_path": str(scene.gaussian_ply_path),
        "gaussian_ply_size_bytes": scene.gaussian_ply_path.stat().st_size,
        "object_count": len(scene.objects),
        "objects_with_bbox": with_bbox,
        "unique_label_count": len(label_hist),
        "top_labels": top_labels,
        "occupancy_image_path": str(scene.occupancy.image_path),
        "occupancy_metadata_keys": sorted(scene.occupancy.metadata.keys()),
        "structure_room_count": scene.structure.room_count,
        "structure_wall_count": scene.structure.wall_count,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
