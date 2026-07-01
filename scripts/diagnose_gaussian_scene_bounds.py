"""Diagnose and optionally prune Gaussians outside the observed scene volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import (
    FrameRecord,
    load_colmap_scene_with_nerfstudio_parser,
    load_nerfstudio_scene,
)
from gsegmenter.mapping import (
    GaussianPruneSpec,
    build_gaussian_prune_mask,
    load_gaussian_cloud,
    load_gaussian_table,
    rgb_to_sh_dc,
    write_gaussian_table,
)
from gsegmenter.render.projection import project_world_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Gaussians that are weakly observed or far outside the camera-covered scene."
    )
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="nerfstudio")
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "ceil"), default="floor")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Use every Nth frame for coverage diagnostics. Smaller is more accurate but slower.",
    )
    parser.add_argument("--min-visible-frames", type=int, default=2)
    parser.add_argument("--radius-percentile", type=float, default=99.5)
    parser.add_argument("--camera-distance-percentile", type=float, default=99.5)
    parser.add_argument("--disable-noise-prune", action="store_true")
    parser.add_argument("--opacity-logit-threshold", type=float, default=-5.0)
    parser.add_argument("--scale-norm-percentile", type=float, default=99.5)
    parser.add_argument("--keep-isolated", action="store_true")
    parser.add_argument("--write-cleaned-ply", action="store_true")
    parser.add_argument("--write-highlight-ply", action="store_true")
    parser.add_argument("--highlight-rgb", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--dim-kept-opacity-scale", type=float, default=0.35)
    return parser.parse_args()


def _load_scene(args: argparse.Namespace):
    if args.scene_format == "nerfstudio":
        return load_nerfstudio_scene(args.dataset_root)
    return load_colmap_scene_with_nerfstudio_parser(
        args.dataset_root,
        downscale_factor=args.downscale_factor,
        downscale_rounding_mode=args.downscale_rounding_mode,
        images_path=args.images_path,
        colmap_path=args.colmap_path,
    )


def _sample_frames(frames: tuple[FrameRecord, ...], frame_stride: int) -> tuple[FrameRecord, ...]:
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    sampled = tuple(frames[::frame_stride])
    if not sampled:
        raise ValueError("No frames were selected for scene-bound diagnostics.")
    return sampled


def _compute_visibility_counts(
    xyz: np.ndarray,
    frames: tuple[FrameRecord, ...],
    intrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Count per-Gaussian positive-depth and image-bounds observations."""

    visible_counts = np.zeros((xyz.shape[0],), dtype=np.uint16)
    in_front_counts = np.zeros((xyz.shape[0],), dtype=np.uint16)
    for frame in frames:
        projection = project_world_points(xyz, intrinsics, frame)
        visible_counts += projection.valid_mask.astype(np.uint16)
        in_front_counts += (projection.depths > 0.0).astype(np.uint16)
    return visible_counts, in_front_counts


def _nearest_camera_distances(xyz: np.ndarray, frames: tuple[FrameRecord, ...]) -> np.ndarray:
    camera_centers = np.asarray([frame.camera_to_world[:3, 3] for frame in frames], dtype=np.float32)
    nearest = np.full((xyz.shape[0],), np.inf, dtype=np.float32)
    chunk_size = 65536
    for start in range(0, xyz.shape[0], chunk_size):
        end = min(start + chunk_size, xyz.shape[0])
        delta = xyz[start:end, None, :] - camera_centers[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        nearest[start:end] = distances.min(axis=1)
    return nearest


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_5": float(np.percentile(values, 99.5)),
    }


def _write_highlight_ply(
    output_path: Path,
    table: np.ndarray,
    header_properties: list[tuple[str, str]],
    remove_mask: np.ndarray,
    *,
    highlight_rgb: tuple[float, float, float],
    dim_kept_opacity_scale: float,
) -> None:
    highlighted = table.copy()
    target_dc = rgb_to_sh_dc(np.asarray(highlight_rgb, dtype=np.float32)).astype(np.float32)
    for channel, value in zip(("f_dc_0", "f_dc_1", "f_dc_2"), target_dc, strict=True):
        if channel not in highlighted.dtype.names:
            raise ValueError(f"Input PLY does not contain required color channel {channel!r}.")
        highlighted[channel][remove_mask] = value
    for property_name in highlighted.dtype.names:
        if property_name.startswith("f_rest_"):
            highlighted[property_name][remove_mask] = np.float32(0.0)
    if "opacity" in highlighted.dtype.names and dim_kept_opacity_scale != 1.0:
        if dim_kept_opacity_scale <= 0.0:
            raise ValueError("--dim-kept-opacity-scale must be positive.")
        highlighted["opacity"][~remove_mask] = highlighted["opacity"][~remove_mask] + np.float32(
            np.log(dim_kept_opacity_scale)
        )
    write_gaussian_table(output_path, highlighted, header_properties)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    table, header_properties = load_gaussian_table(args.ply_path)
    cloud = load_gaussian_cloud(args.ply_path)
    scene = _load_scene(args)
    frames = _sample_frames(scene.frames, args.frame_stride)
    xyz = cloud.xyz.astype(np.float32)

    visible_counts, in_front_counts = _compute_visibility_counts(xyz, frames, scene.intrinsics)
    radius_center = np.median(xyz, axis=0, keepdims=True)
    radius = np.linalg.norm(xyz - radius_center, axis=1).astype(np.float32)
    nearest_camera_distance = _nearest_camera_distances(xyz, frames)

    radius_cutoff = float(np.percentile(radius, args.radius_percentile))
    camera_distance_cutoff = float(np.percentile(nearest_camera_distance, args.camera_distance_percentile))
    weak_visibility = visible_counts < int(args.min_visible_frames)
    radius_outlier = radius > radius_cutoff
    camera_distance_outlier = nearest_camera_distance > camera_distance_cutoff
    scene_outside_candidate = weak_visibility & (radius_outlier | camera_distance_outlier)

    if args.disable_noise_prune:
        noise_remove = np.zeros_like(scene_outside_candidate, dtype=bool)
    else:
        prune_spec = GaussianPruneSpec(
            opacity_logit_threshold=args.opacity_logit_threshold,
            scale_norm_percentile=args.scale_norm_percentile,
            remove_isolated=not args.keep_isolated,
            remove_low_opacity=True,
            remove_extreme_scales=True,
            remove_radius_outliers=False,
        )
        noise_remove = ~build_gaussian_prune_mask(cloud, prune_spec)

    remove_mask = scene_outside_candidate | noise_remove
    keep_mask = ~remove_mask

    np.savez_compressed(
        args.output_root / "scene_bounds_diagnostics.npz",
        keep_mask=keep_mask,
        remove_mask=remove_mask,
        scene_outside_candidate=scene_outside_candidate,
        noise_remove=noise_remove,
        visible_counts=visible_counts,
        in_front_counts=in_front_counts,
        radius=radius,
        nearest_camera_distance=nearest_camera_distance,
    )

    report = {
        "gaussian_count": int(xyz.shape[0]),
        "sampled_frame_count": int(len(frames)),
        "frame_stride": int(args.frame_stride),
        "min_visible_frames": int(args.min_visible_frames),
        "radius_percentile": float(args.radius_percentile),
        "radius_cutoff": radius_cutoff,
        "camera_distance_percentile": float(args.camera_distance_percentile),
        "camera_distance_cutoff": camera_distance_cutoff,
        "visible_count_quantiles": _quantiles(visible_counts.astype(np.float32)),
        "radius_quantiles": _quantiles(radius),
        "nearest_camera_distance_quantiles": _quantiles(nearest_camera_distance),
        "weak_visibility_count": int(np.count_nonzero(weak_visibility)),
        "radius_outlier_count": int(np.count_nonzero(radius_outlier)),
        "camera_distance_outlier_count": int(np.count_nonzero(camera_distance_outlier)),
        "scene_outside_candidate_count": int(np.count_nonzero(scene_outside_candidate)),
        "noise_remove_count": int(np.count_nonzero(noise_remove)),
        "remove_count": int(np.count_nonzero(remove_mask)),
        "keep_count": int(np.count_nonzero(keep_mask)),
    }
    (args.output_root / "scene_bounds_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if args.write_cleaned_ply:
        write_gaussian_table(args.output_root / "cleaned_splat.ply", table[keep_mask], header_properties)
    if args.write_highlight_ply:
        _write_highlight_ply(
            args.output_root / "scene_outside_candidates_highlight.ply",
            table,
            header_properties,
            remove_mask,
            highlight_rgb=tuple(float(v) for v in args.highlight_rgb),
            dim_kept_opacity_scale=float(args.dim_kept_opacity_scale),
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
