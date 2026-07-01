"""Export grouped Gaussians as a colored point cloud PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.gaussian_io import load_gaussian_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Gaussian groups as a colored PLY.")
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--object-ids", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--drop-unknown", action="store_true")
    parser.add_argument("--skip-largest-n", type=int, default=0)
    parser.add_argument("--min-group-size", type=int, default=0)
    parser.add_argument("--include-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--highlight-red", action="store_true")
    return parser.parse_args()


def _color_from_object_id(object_id: int) -> tuple[int, int, int]:
    """Generate a deterministic RGB color from a global object id."""

    if object_id < 0:
        return 96, 96, 96
    seed = (object_id * 1103515245 + 12345) & 0x7FFFFFFF
    red = 64 + (seed & 0x7F)
    green = 64 + ((seed >> 7) & 0x7F)
    blue = 64 + ((seed >> 14) & 0x7F)
    return red, green, blue


def _load_keep_object_ids(
    groups_json: Path | None,
    skip_largest_n: int,
    min_group_size: int,
    include_object_ids: list[int] | None,
    exclude_object_ids: list[int] | None,
) -> set[int] | None:
    keep_object_ids: set[int] | None = None
    if groups_json is not None:
        payload = json.loads(groups_json.read_text(encoding="utf-8"))
        groups = sorted(payload["groups"], key=lambda group: int(group["gaussian_count"]), reverse=True)
        skipped = {int(group["global_object_id"]) for group in groups[:skip_largest_n]} if skip_largest_n > 0 else set()
        keep_object_ids = {
            int(group["global_object_id"])
            for group in groups
            if int(group["gaussian_count"]) >= min_group_size and int(group["global_object_id"]) not in skipped
        }

    if include_object_ids is not None:
        explicit = {int(object_id) for object_id in include_object_ids}
        keep_object_ids = explicit if keep_object_ids is None else (keep_object_ids & explicit)
    if exclude_object_ids is not None:
        excluded = {int(object_id) for object_id in exclude_object_ids}
        if keep_object_ids is None:
            keep_object_ids = None
        else:
            keep_object_ids -= excluded
    return keep_object_ids


def _write_ascii_ply(
    xyz: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
) -> None:
    """Write an ASCII point cloud PLY with vertex RGB values."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, colors, strict=True):
            handle.write(
                f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> int:
    args = parse_args()
    cloud = load_gaussian_cloud(args.ply_path)
    object_ids = np.load(args.object_ids)
    if len(object_ids) != cloud.vertex_count:
        raise ValueError(
            f"Object id count {len(object_ids)} does not match Gaussian count {cloud.vertex_count}"
        )

    xyz = cloud.xyz
    keep_object_ids = _load_keep_object_ids(
        args.groups_json,
        args.skip_largest_n,
        args.min_group_size,
        args.include_object_ids,
        args.exclude_object_ids,
    )
    if keep_object_ids is None:
        selected_mask = np.ones_like(object_ids, dtype=bool)
        if args.exclude_object_ids is not None:
            selected_mask &= ~np.isin(object_ids, np.asarray(args.exclude_object_ids, dtype=np.int64))
    else:
        selected_mask = np.isin(object_ids, np.asarray(sorted(keep_object_ids), dtype=np.int64))

    if args.drop_unknown:
        selected_mask &= object_ids >= 0

    if args.max_points is not None and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(42)
        if args.show_context and np.any(selected_mask):
            selected_indices = np.flatnonzero(selected_mask)
            unselected_indices = np.flatnonzero(~selected_mask)
            selected_budget = min(len(selected_indices), max(1, args.max_points // 2))
            context_budget = min(len(unselected_indices), max(0, args.max_points - selected_budget))
            sampled_selected = (
                rng.choice(selected_indices, size=selected_budget, replace=False)
                if selected_budget < len(selected_indices)
                else selected_indices
            )
            sampled_context = (
                rng.choice(unselected_indices, size=context_budget, replace=False)
                if context_budget < len(unselected_indices)
                else unselected_indices
            )
            sample_indices = np.concatenate([sampled_selected, sampled_context])
        else:
            candidate_indices = np.flatnonzero(selected_mask)
            if candidate_indices.size == 0:
                raise ValueError("No Gaussians remain after applying group filters.")
            sample_size = min(args.max_points, candidate_indices.size)
            sample_indices = rng.choice(candidate_indices, size=sample_size, replace=False)
        xyz = xyz[sample_indices]
        object_ids = object_ids[sample_indices]
        selected_mask = selected_mask[sample_indices]
    elif not args.show_context:
        xyz = xyz[selected_mask]
        object_ids = object_ids[selected_mask]
        selected_mask = np.ones_like(object_ids, dtype=bool)

    if args.show_context:
        colors = np.full((object_ids.shape[0], 3), 96, dtype=np.uint8)
        if args.highlight_red:
            colors[selected_mask] = np.asarray([255, 48, 48], dtype=np.uint8)
        else:
            colors[selected_mask] = np.asarray(
                [_color_from_object_id(int(object_id)) for object_id in object_ids[selected_mask]],
                dtype=np.uint8,
            )
    else:
        colors = np.asarray([_color_from_object_id(int(object_id)) for object_id in object_ids], dtype=np.uint8)
    _write_ascii_ply(xyz, colors, args.output_path)
    print(f"Wrote colored group point cloud to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
