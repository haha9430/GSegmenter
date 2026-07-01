"""Prune noisy gaussians from an exported `splat.ply` and optional sidecar arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import (
    GaussianPruneSpec,
    build_gaussian_prune_mask,
    filter_gaussian_sidecar,
    load_gaussian_cloud,
    load_gaussian_table,
    write_gaussian_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune noisy gaussians from an exported PLY.")
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--sidecar-path", type=Path, default=None)
    parser.add_argument("--output-sidecar", type=Path, default=None)
    parser.add_argument("--opacity-logit-threshold", type=float, default=-5.0)
    parser.add_argument("--scale-norm-percentile", type=float, default=99.5)
    parser.add_argument("--radius-percentile", type=float, default=99.9)
    parser.add_argument("--radius-multiplier", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--isolated-min-neighbor-count", type=int, default=2)
    parser.add_argument("--keep-radius-outliers", action="store_true")
    parser.add_argument("--keep-isolated", action="store_true")
    parser.add_argument("--keep-low-opacity", action="store_true")
    parser.add_argument("--keep-extreme-scales", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table, header_properties = load_gaussian_table(args.ply_path)
    cloud = load_gaussian_cloud(args.ply_path)
    spec = GaussianPruneSpec(
        opacity_logit_threshold=args.opacity_logit_threshold,
        scale_norm_percentile=args.scale_norm_percentile,
        radius_percentile=args.radius_percentile,
        radius_multiplier=args.radius_multiplier,
        voxel_size=args.voxel_size,
        isolated_min_neighbor_count=args.isolated_min_neighbor_count,
        remove_isolated=not args.keep_isolated,
        remove_low_opacity=not args.keep_low_opacity,
        remove_extreme_scales=not args.keep_extreme_scales,
        remove_radius_outliers=not args.keep_radius_outliers,
    )
    keep_mask = build_gaussian_prune_mask(cloud, spec)
    filtered_table = table[keep_mask]
    write_gaussian_table(args.output_ply, filtered_table, header_properties)
    removed = int((~keep_mask).sum())
    print(f"Pruned {removed} / {keep_mask.shape[0]} gaussians")
    print(f"Wrote filtered PLY to {args.output_ply}")

    if args.sidecar_path is not None:
        output_sidecar = args.output_sidecar or args.output_ply.with_name(args.output_ply.stem + "_identity.npy")
        try:
            filter_gaussian_sidecar(args.sidecar_path, keep_mask, output_sidecar)
        except ValueError as error:
            print(f"Skipped sidecar filtering: {error}")
            return 0
        print(f"Wrote filtered sidecar to {output_sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
