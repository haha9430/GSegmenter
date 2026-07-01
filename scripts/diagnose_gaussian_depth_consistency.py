"""Find Gaussians that repeatedly sit behind monocularly estimated visible surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import (
    load_colmap_scene_with_nerfstudio_parser,
    load_nerfstudio_scene,
)
from gsegmenter.mapping import load_gaussian_cloud, load_gaussian_table, rgb_to_sh_dc, write_gaussian_table
from gsegmenter.render.projection import project_world_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare projected Gaussian depths with aligned monocular depth maps."
    )
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="nerfstudio")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "ceil"), default="floor")
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--fit-max-points", type=int, default=30000)
    parser.add_argument("--trim-quantile", type=float, default=0.80)
    parser.add_argument("--behind-margin-ratio", type=float, default=0.20)
    parser.add_argument("--behind-min-margin", type=float, default=0.05)
    parser.add_argument("--min-observed-frames", type=int, default=3)
    parser.add_argument("--behind-ratio-threshold", type=float, default=0.60)
    parser.add_argument("--write-highlight-ply", action="store_true")
    parser.add_argument("--write-cleaned-ply", action="store_true")
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


def _fit_affine_robust(predicted: np.ndarray, target: np.ndarray, trim_quantile: float) -> tuple[float, float, float]:
    """Fit `target ~= scale * predicted + shift` with residual trimming."""

    finite = np.isfinite(predicted) & np.isfinite(target)
    x = predicted[finite].astype(np.float64)
    y = target[finite].astype(np.float64)
    if x.size < 32:
        return 1.0, 0.0, float("inf")

    keep = np.ones_like(x, dtype=bool)
    scale = 1.0
    shift = 0.0
    trim_quantile = float(np.clip(trim_quantile, 0.1, 1.0))
    for _ in range(4):
        design = np.stack([x[keep], np.ones(int(np.count_nonzero(keep)))], axis=1)
        scale, shift = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = np.abs((scale * x + shift) - y)
        cutoff = float(np.quantile(residual, trim_quantile))
        keep = residual <= cutoff
        if np.count_nonzero(keep) < 32:
            break
    residual = np.abs((scale * x + shift) - y)
    return float(scale), float(shift), float(np.median(residual[keep]))


def _front_surface_samples(
    image_points: np.ndarray,
    depths: np.ndarray,
    valid_mask: np.ndarray,
    depth_map: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the front-most Gaussian depth per occupied pixel for affine fitting."""

    indices = np.flatnonzero(valid_mask)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    px = np.floor(image_points[indices, 0]).astype(np.int64)
    py = np.floor(image_points[indices, 1]).astype(np.int64)
    height, width = depth_map.shape
    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    indices = indices[in_bounds]
    px = px[in_bounds]
    py = py[in_bounds]
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    linear = py * width + px
    order = np.lexsort((depths[indices], linear))
    sorted_linear = linear[order]
    sorted_indices = indices[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    front_indices = sorted_indices[first]
    front_px = px[order][first]
    front_py = py[order][first]

    if front_indices.size > max_points:
        rng = np.random.default_rng(42)
        chosen = rng.choice(front_indices.size, size=max_points, replace=False)
        front_indices = front_indices[chosen]
        front_px = front_px[chosen]
        front_py = front_py[chosen]

    return depth_map[front_py, front_px].astype(np.float32), depths[front_indices].astype(np.float32)


def _write_highlight_ply(
    output_path: Path,
    table: np.ndarray,
    header_properties: list[tuple[str, str]],
    remove_mask: np.ndarray,
    dim_kept_opacity_scale: float,
) -> None:
    highlighted = table.copy()
    target_dc = rgb_to_sh_dc(np.asarray([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    for channel, value in zip(("f_dc_0", "f_dc_1", "f_dc_2"), target_dc, strict=True):
        highlighted[channel][remove_mask] = value
    for property_name in highlighted.dtype.names:
        if property_name.startswith("f_rest_"):
            highlighted[property_name][remove_mask] = np.float32(0.0)
    if "opacity" in highlighted.dtype.names and dim_kept_opacity_scale != 1.0:
        highlighted["opacity"][~remove_mask] = highlighted["opacity"][~remove_mask] + np.float32(
            np.log(float(dim_kept_opacity_scale))
        )
    write_gaussian_table(output_path, highlighted, header_properties)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    table, header_properties = load_gaussian_table(args.ply_path)
    cloud = load_gaussian_cloud(args.ply_path)
    scene = _load_scene(args)
    frames = tuple(scene.frames[:: args.frame_stride])
    if args.limit_frames is not None:
        frames = frames[: args.limit_frames]
    if not frames:
        raise ValueError("No frames selected for depth consistency diagnostics.")

    observed_counts = np.zeros((cloud.vertex_count,), dtype=np.uint16)
    behind_counts = np.zeros((cloud.vertex_count,), dtype=np.uint16)
    frame_reports: list[dict[str, object]] = []

    for frame in frames:
        depth_path = args.depth_root / frame.file_path.stem / "depth.npy"
        if not depth_path.exists():
            continue
        depth_map = np.load(depth_path).astype(np.float32)
        projection = project_world_points(cloud.xyz, scene.intrinsics, frame)
        if depth_map.shape != (scene.intrinsics.height, scene.intrinsics.width):
            raise ValueError(
                f"Depth map shape {depth_map.shape} does not match intrinsics "
                f"{(scene.intrinsics.height, scene.intrinsics.width)} for {depth_path}"
            )

        fit_pred, fit_target = _front_surface_samples(
            projection.image_points,
            projection.depths,
            projection.valid_mask,
            depth_map,
            args.fit_max_points,
        )
        scale, shift, median_residual = _fit_affine_robust(fit_pred, fit_target, args.trim_quantile)

        indices = np.flatnonzero(projection.valid_mask)
        px = np.floor(projection.image_points[indices, 0]).astype(np.int64)
        py = np.floor(projection.image_points[indices, 1]).astype(np.int64)
        predicted_surface_depth = scale * depth_map[py, px].astype(np.float32) + shift
        margin = np.maximum(
            float(args.behind_min_margin),
            np.abs(predicted_surface_depth) * float(args.behind_margin_ratio),
        )
        behind = projection.depths[indices] > (predicted_surface_depth + margin)
        observed_counts[indices] += 1
        behind_counts[indices[behind]] += 1
        frame_reports.append(
            {
                "frame_index": int(frame.index),
                "image_stem": frame.file_path.stem,
                "valid_projection_count": int(indices.size),
                "fit_point_count": int(fit_pred.size),
                "affine_scale": scale,
                "affine_shift": shift,
                "fit_median_abs_residual": median_residual,
                "behind_count": int(np.count_nonzero(behind)),
            }
        )

    behind_ratio = np.zeros((cloud.vertex_count,), dtype=np.float32)
    observed = observed_counts > 0
    behind_ratio[observed] = behind_counts[observed].astype(np.float32) / observed_counts[observed].astype(np.float32)
    remove_mask = (
        (observed_counts >= int(args.min_observed_frames))
        & (behind_ratio >= float(args.behind_ratio_threshold))
    )
    keep_mask = ~remove_mask

    np.savez_compressed(
        args.output_root / "depth_consistency_diagnostics.npz",
        keep_mask=keep_mask,
        remove_mask=remove_mask,
        observed_counts=observed_counts,
        behind_counts=behind_counts,
        behind_ratio=behind_ratio,
    )
    report = {
        "gaussian_count": int(cloud.vertex_count),
        "processed_frame_count": len(frame_reports),
        "frame_stride": int(args.frame_stride),
        "min_observed_frames": int(args.min_observed_frames),
        "behind_ratio_threshold": float(args.behind_ratio_threshold),
        "remove_count": int(np.count_nonzero(remove_mask)),
        "keep_count": int(np.count_nonzero(keep_mask)),
        "observed_count_quantiles": {
            "p50": float(np.percentile(observed_counts, 50)),
            "p90": float(np.percentile(observed_counts, 90)),
            "p99": float(np.percentile(observed_counts, 99)),
        },
        "behind_ratio_quantiles_observed": {
            "p50": float(np.percentile(behind_ratio[observed], 50)) if np.any(observed) else 0.0,
            "p90": float(np.percentile(behind_ratio[observed], 90)) if np.any(observed) else 0.0,
            "p99": float(np.percentile(behind_ratio[observed], 99)) if np.any(observed) else 0.0,
        },
        "frames": frame_reports,
    }
    (args.output_root / "depth_consistency_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if args.write_highlight_ply:
        _write_highlight_ply(
            args.output_root / "depth_inconsistent_highlight.ply",
            table,
            header_properties,
            remove_mask,
            float(args.dim_kept_opacity_scale),
        )
    if args.write_cleaned_ply:
        write_gaussian_table(args.output_root / "depth_cleaned_splat.ply", table[keep_mask], header_properties)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
