"""Overlay projected Gaussian groups on a source image frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud
from gsegmenter.render.projection import project_world_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project Gaussian groups onto a frame.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--object-ids", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="nerfstudio")
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="floor")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--show-points", action="store_true")
    parser.add_argument("--boundary-color", type=str, default="#ff2d2d")
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--boundary-dilate", type=int, default=2)
    parser.add_argument("--skip-largest-n", type=int, default=0)
    parser.add_argument("--min-group-size", type=int, default=0)
    parser.add_argument("--include-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-object-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def _color_from_object_id(object_id: int) -> tuple[float, float, float]:
    """Generate a deterministic RGB color in matplotlib range."""

    seed = (object_id * 1103515245 + 12345) & 0x7FFFFFFF
    red = 64 + (seed & 0x7F)
    green = 64 + ((seed >> 7) & 0x7F)
    blue = 64 + ((seed >> 14) & 0x7F)
    return red / 255.0, green / 255.0, blue / 255.0


def _render_group_boundary(
    image_shape: tuple[int, int, int],
    points: np.ndarray,
    dilate_radius: int,
    boundary_width: int,
) -> np.ndarray:
    """Rasterize projected points into a binary boundary mask.

    The grouping output is sparse point support rather than dense masks, so we first
    dilate projected support points, then extract the outer edge. This yields a
    stable outline that is easier to read than raw scatter points.
    """

    height, width = image_shape[:2]
    occupancy = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(occupancy)
    radius = max(1, dilate_radius)
    for point_x, point_y in points:
        center_x = int(round(float(point_x)))
        center_y = int(round(float(point_y)))
        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            continue
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=255,
        )

    dilated = occupancy.filter(ImageFilter.MaxFilter(size=radius * 2 + 1))
    eroded = dilated.filter(ImageFilter.MinFilter(size=radius * 2 + 1))
    boundary = np.asarray(dilated, dtype=np.uint8) > np.asarray(eroded, dtype=np.uint8)
    if boundary_width > 1:
        boundary_image = Image.fromarray(boundary.astype(np.uint8) * 255, mode="L")
        expanded = boundary_image.filter(ImageFilter.MaxFilter(size=boundary_width * 2 + 1))
        boundary = np.asarray(expanded, dtype=np.uint8) > 0
    return boundary


def main() -> int:
    args = parse_args()
    if args.scene_format == "colmap":
        scene = load_colmap_scene_with_nerfstudio_parser(
            args.dataset_root,
            downscale_factor=args.downscale_factor,
            downscale_rounding_mode=args.downscale_rounding_mode,
            images_path=args.images_path,
            colmap_path=args.colmap_path,
        )
    else:
        scene = load_nerfstudio_scene(args.dataset_root)
    cloud = load_gaussian_cloud(args.ply_path)
    group_payload = json.loads(args.groups_json.read_text(encoding="utf-8"))
    object_ids = np.load(args.object_ids)

    if args.frame_index < 0 or args.frame_index >= len(scene.frames):
        raise ValueError(f"Frame index {args.frame_index} is out of range for {len(scene.frames)} frames")

    frame = scene.frames[args.frame_index]
    projection = project_world_points(cloud.xyz, scene.intrinsics, frame)
    with Image.open(frame.file_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    top_groups = sorted(
        group_payload["groups"],
        key=lambda entry: int(entry["gaussian_count"]),
        reverse=True,
    )
    if args.skip_largest_n > 0:
        top_groups = top_groups[args.skip_largest_n :]
    if args.min_group_size > 0:
        top_groups = [
            group for group in top_groups if int(group["gaussian_count"]) >= args.min_group_size
        ]
    if args.include_object_ids is not None:
        included = {int(object_id) for object_id in args.include_object_ids}
        top_groups = [group for group in top_groups if int(group["global_object_id"]) in included]
    if args.exclude_object_ids is not None:
        excluded = {int(object_id) for object_id in args.exclude_object_ids}
        top_groups = [group for group in top_groups if int(group["global_object_id"]) not in excluded]
    top_groups = top_groups[: args.top_k]

    fig, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(rgb)
    axis.set_title(f"Frame {args.frame_index:04d} group projection")
    axis.axis("off")

    for group in top_groups:
        object_id = int(group["global_object_id"])
        group_mask = (object_ids == object_id) & projection.valid_mask
        if not np.any(group_mask):
            continue
        points = projection.image_points[group_mask]
        boundary_mask = _render_group_boundary(
            rgb.shape,
            points,
            args.boundary_dilate,
            args.boundary_width,
        )
        boundary_overlay = np.zeros((*boundary_mask.shape, 4), dtype=np.float32)
        boundary_overlay[boundary_mask, :3] = np.asarray(
            plt.matplotlib.colors.to_rgb(args.boundary_color),
            dtype=np.float32,
        )
        boundary_overlay[boundary_mask, 3] = 1.0
        axis.imshow(boundary_overlay)

        if args.show_points:
            color = _color_from_object_id(object_id)
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=args.point_size,
                c=[color],
                alpha=args.alpha,
                label=f"id={object_id} ({group['gaussian_count']})",
                linewidths=0.0,
            )
        else:
            axis.plot(
                [],
                [],
                color=args.boundary_color,
                linewidth=max(1, args.boundary_width),
                label=f"id={object_id} ({group['gaussian_count']})",
            )

    if axis.collections:
        axis.legend(loc="upper right", fontsize=8, framealpha=0.8)
    elif axis.lines:
        axis.legend(loc="upper right", fontsize=8, framealpha=0.8)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote group projection overlay to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
