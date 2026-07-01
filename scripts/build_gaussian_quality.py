"""Build Gaussian quality sidecars for SAM-to-3D vote lifting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import (
    GaussianPruneSpec,
    GaussianQualitySpec,
    build_gaussian_prune_mask,
    build_gaussian_quality_scores,
    compute_gaussian_noise_report,
    load_gaussian_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gaussian quality arrays for identity lifting.")
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--opacity-alpha-floor", type=float, default=0.05)
    parser.add_argument("--opacity-logit-threshold", type=float, default=-5.0)
    parser.add_argument("--scale-norm-percentile", type=float, default=99.5)
    parser.add_argument("--radius-percentile", type=float, default=99.9)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--isolated-min-neighbor-count", type=int, default=2)
    parser.add_argument("--keep-isolated", action="store_true")
    parser.add_argument("--keep-low-opacity", action="store_true")
    parser.add_argument("--keep-extreme-scales", action="store_true")
    parser.add_argument("--remove-radius-outliers", action="store_true")
    return parser.parse_args()


def _count(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def main() -> int:
    args = parse_args()
    cloud = load_gaussian_cloud(args.ply_path)
    quality_spec = GaussianQualitySpec(
        opacity_alpha_floor=args.opacity_alpha_floor,
        scale_norm_percentile=args.scale_norm_percentile,
        radius_percentile=args.radius_percentile,
        voxel_size=args.voxel_size,
        isolated_min_neighbor_count=args.isolated_min_neighbor_count,
    )
    quality, diagnostics = build_gaussian_quality_scores(cloud, quality_spec)

    prune_spec = GaussianPruneSpec(
        opacity_logit_threshold=args.opacity_logit_threshold,
        scale_norm_percentile=args.scale_norm_percentile,
        radius_percentile=args.radius_percentile,
        voxel_size=args.voxel_size,
        isolated_min_neighbor_count=args.isolated_min_neighbor_count,
        remove_isolated=not args.keep_isolated,
        remove_low_opacity=not args.keep_low_opacity,
        remove_extreme_scales=not args.keep_extreme_scales,
        remove_radius_outliers=args.remove_radius_outliers,
    )
    valid_mask = build_gaussian_prune_mask(cloud, prune_spec)
    report = compute_gaussian_noise_report(
        cloud,
        opacity_logit_threshold=args.opacity_logit_threshold,
        scale_norm_percentile=args.scale_norm_percentile,
        voxel_size=args.voxel_size,
        isolated_min_neighbor_count=args.isolated_min_neighbor_count,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    quality_path = args.output_root / "gaussian_quality.npy"
    valid_path = args.output_root / "valid_gaussian_mask.npy"
    diagnostics_path = args.output_root / "gaussian_quality_diagnostics.npz"
    report_path = args.output_root / "gaussian_quality_report.json"
    np.save(quality_path, quality)
    np.save(valid_path, valid_mask)
    np.savez_compressed(diagnostics_path, **diagnostics)

    payload = {
        "ply_path": str(args.ply_path),
        "gaussian_count": int(cloud.vertex_count),
        "valid_count": int(valid_mask.sum()),
        "invalid_count": int((~valid_mask).sum()),
        "quality_quantiles": {
            "p01": float(np.percentile(quality, 1)) if quality.size else 0.0,
            "p10": float(np.percentile(quality, 10)) if quality.size else 0.0,
            "p50": float(np.percentile(quality, 50)) if quality.size else 0.0,
            "p90": float(np.percentile(quality, 90)) if quality.size else 0.0,
            "p99": float(np.percentile(quality, 99)) if quality.size else 0.0,
        },
        "diagnostic_counts": {name: _count(mask) for name, mask in diagnostics.items()},
        "noise_report": report.to_dict(),
        "outputs": {
            "quality": str(quality_path),
            "valid_mask": str(valid_path),
            "diagnostics": str(diagnostics_path),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
