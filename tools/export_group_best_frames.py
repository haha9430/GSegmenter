"""Find the best visible frame for each grouped object and export overlays."""

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

from gsegmenter.data.nerfstudio_scene import load_nerfstudio_scene
from gsegmenter.mapping import find_best_frame_for_group, load_gaussian_cloud, select_group_ids
from gsegmenter.render.projection import project_world_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export best-frame overlays for grouped Gaussian objects."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--object-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--skip-largest-n", type=int, default=0)
    parser.add_argument("--min-group-size", type=int, default=0)
    parser.add_argument("--include-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--boundary-color", type=str, default="#ff2d2d")
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--boundary-dilate", type=int, default=2)
    parser.add_argument("--show-points", action="store_true")
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def _render_group_boundary(
    image_shape: tuple[int, int, int],
    points: np.ndarray,
    dilate_radius: int,
    boundary_width: int,
) -> np.ndarray:
    """Rasterize projected support points into a readable 2D outline."""

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


def _save_overlay(
    rgb: np.ndarray,
    projected_points: np.ndarray,
    object_id: int,
    visible_count: int,
    gaussian_count: int,
    frame_index: int,
    output_path: Path,
    *,
    boundary_color: str,
    boundary_width: int,
    boundary_dilate: int,
    show_points: bool,
    point_size: float,
    alpha: float,
) -> None:
    """Save a single group overlay image."""

    fig, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(rgb)
    axis.set_title(
        f"object={object_id} frame={frame_index:04d} visible={visible_count}/{gaussian_count}"
    )
    axis.axis("off")

    boundary_mask = _render_group_boundary(rgb.shape, projected_points, boundary_dilate, boundary_width)
    boundary_overlay = np.zeros((*boundary_mask.shape, 4), dtype=np.float32)
    boundary_overlay[boundary_mask, :3] = np.asarray(
        plt.matplotlib.colors.to_rgb(boundary_color),
        dtype=np.float32,
    )
    boundary_overlay[boundary_mask, 3] = 1.0
    axis.imshow(boundary_overlay)

    if show_points:
        axis.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            s=point_size,
            c=[plt.matplotlib.colors.to_rgb(boundary_color)],
            alpha=alpha,
            linewidths=0.0,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    scene = load_nerfstudio_scene(args.dataset_root)
    cloud = load_gaussian_cloud(args.ply_path)
    group_payload = json.loads(args.groups_json.read_text(encoding="utf-8"))
    object_ids = np.load(args.object_ids)

    selected_ids = select_group_ids(
        group_payload["groups"],
        top_k=args.top_k,
        skip_largest_n=args.skip_largest_n,
        min_group_size=args.min_group_size,
        include_object_ids=args.include_object_ids,
        exclude_object_ids=args.exclude_object_ids,
    )
    if not selected_ids:
        raise ValueError("No groups remain after applying the requested filters.")

    summaries: list[dict[str, object]] = []
    for object_id in selected_ids:
        visibility = find_best_frame_for_group(scene, cloud.xyz, object_ids, object_id)
        object_mask = object_ids == object_id
        frame = scene.frames[visibility.best_frame_index]
        projection = project_world_points(cloud.xyz[object_mask], scene.intrinsics, frame)
        visible_points = projection.image_points[projection.valid_mask]
        with Image.open(frame.file_path) as image:
            rgb = np.asarray(image.convert("RGB"))

        image_path = args.output_root / f"object_{object_id}_frame_{visibility.best_frame_index:04d}.png"
        _save_overlay(
            rgb,
            visible_points,
            object_id,
            visibility.best_visible_count,
            visibility.gaussian_count,
            visibility.best_frame_index,
            image_path,
            boundary_color=args.boundary_color,
            boundary_width=args.boundary_width,
            boundary_dilate=args.boundary_dilate,
            show_points=args.show_points,
            point_size=args.point_size,
            alpha=args.alpha,
        )

        summaries.append(
            {
                "object_id": visibility.object_id,
                "gaussian_count": visibility.gaussian_count,
                "best_frame_index": visibility.best_frame_index,
                "best_visible_count": visibility.best_visible_count,
                "visibility_ratio": visibility.visibility_ratio,
                "overlay_path": str(image_path),
            }
        )

    summary_path = args.output_root / "best_frame_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"groups": summaries}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote best-frame overlays to {args.output_root}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
