"""Build Gaussian object groups from InteriorGS annotation boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import load_interiorgs_scene
from gsegmenter.mapping import (
    assign_gaussians_to_interiorgs_objects,
    load_gaussian_cloud,
    save_interiorgs_group_outputs,
    summarize_interiorgs_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign InteriorGS object labels to uncompressed Gaussian centers."
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        required=True,
        help="InteriorGS scene directory containing labels.json and 3dgs_uncompressed.ply.",
    )
    parser.add_argument(
        "--ply-path",
        type=Path,
        default=None,
        help="Override path to the uncompressed Gaussian PLY. Defaults to <scene-root>/3dgs_uncompressed.ply.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where gaussian_object_ids.npy and gaussian_groups.json will be written.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Numerical margin for box inclusion checks.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    scene = load_interiorgs_scene(args.scene_root)
    ply_path = args.ply_path or (args.scene_root / "3dgs_uncompressed.ply")
    if not ply_path.exists():
        raise FileNotFoundError(
            f"Expected uncompressed Gaussian PLY at {ply_path}. "
            "Run scripts/convert_interiorgs_ply.py first."
        )

    cloud = load_gaussian_cloud(ply_path)
    gaussian_object_ids, boxes = assign_gaussians_to_interiorgs_objects(
        cloud.xyz,
        scene.objects,
        epsilon=args.epsilon,
    )
    groups = summarize_interiorgs_groups(gaussian_object_ids, cloud.xyz, boxes)
    save_interiorgs_group_outputs(gaussian_object_ids, groups, args.output_root)

    summary = {
        "gaussian_count": int(cloud.vertex_count),
        "box_count": len(boxes),
        "group_count": len(groups),
        "assigned_gaussians": int((gaussian_object_ids >= 0).sum()),
        "unknown_gaussians": int((gaussian_object_ids < 0).sum()),
        "top_groups": [
            {
                "object_id": group.object_id,
                "label": group.label,
                "gaussian_count": group.gaussian_count,
            }
            for group in sorted(groups, key=lambda item: item.gaussian_count, reverse=True)[:10]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
