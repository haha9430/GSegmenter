"""Bounded source-region cleanup for a moved or removed Gaussian object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.editor.repair import cleanup_source_region_appearance
from gsegmenter.mapping.gaussian_io import load_gaussian_table, write_gaussian_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce source-region residuals after moving an object."
    )
    parser.add_argument("--ply-path", type=Path, required=True, help="Input edited Gaussian PLY.")
    parser.add_argument("--object-ids", type=Path, required=True, help="gaussian_object_ids.npy aligned with the PLY.")
    parser.add_argument("--groups-json", type=Path, required=True, help="Group summary JSON containing bbox info.")
    parser.add_argument("--target-object-id", type=int, required=True, help="Object id whose original region will be cleaned.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output Gaussian PLY after cleanup.")
    parser.add_argument(
        "--mode",
        choices=("blend", "opacity_only"),
        default="blend",
        help="Cleanup strategy. opacity_only is more conservative on reflective backgrounds.",
    )
    parser.add_argument("--shell-margin", type=float, default=0.12, help="Margin around the source bbox for background sampling.")
    parser.add_argument("--color-blend", type=float, default=0.85, help="Blend factor toward nearby background appearance.")
    parser.add_argument("--opacity-scale", type=float, default=0.85, help="Opacity logit scale for cleaned source-region gaussians.")
    parser.add_argument("--keep-high-order-sh", action="store_true", help="Preserve f_rest_* instead of zeroing them in the cleaned region.")
    return parser


def _load_group_record(groups_json: Path, target_object_id: int) -> dict:
    payload = json.loads(groups_json.read_text(encoding="utf-8"))
    for group in payload.get("groups", []):
        group_object_id = int(group.get("object_id", group.get("global_object_id", -1)))
        if group_object_id == int(target_object_id):
            return group
    raise KeyError(f"Could not find object id {target_object_id} in {groups_json}")


def main() -> int:
    args = build_parser().parse_args()
    table, header_properties = load_gaussian_table(args.ply_path)
    object_ids = np.load(args.object_ids)
    group = _load_group_record(args.groups_json, args.target_object_id)
    source_bbox_min_xyz = np.asarray(group["annotation_bbox_min_xyz"], dtype=np.float32)
    source_bbox_max_xyz = np.asarray(group["annotation_bbox_max_xyz"], dtype=np.float32)

    repaired, summary = cleanup_source_region_appearance(
        table,
        object_ids,
        args.target_object_id,
        source_bbox_min_xyz,
        source_bbox_max_xyz,
        shell_margin=float(args.shell_margin),
        color_blend=float(args.color_blend),
        opacity_scale=float(args.opacity_scale),
        zero_high_order_sh=not args.keep_high_order_sh,
        mode=args.mode,
    )
    write_gaussian_table(args.output_path, repaired, header_properties)
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "target_object_id": int(args.target_object_id),
                "mode": args.mode,
                "inner_count": summary.inner_count,
                "shell_count": summary.shell_count,
                "color_blend": summary.color_blend,
                "opacity_scale": summary.opacity_scale,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
