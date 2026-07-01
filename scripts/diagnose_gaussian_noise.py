"""Summarize noise characteristics of an exported Gaussian PLY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import compute_gaussian_noise_report, load_gaussian_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose noise in an exported Gaussian PLY.")
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--opacity-logit-threshold", type=float, default=-5.0)
    parser.add_argument("--scale-norm-percentile", type=float, default=99.5)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--isolated-min-neighbor-count", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cloud = load_gaussian_cloud(args.ply_path)
    report = compute_gaussian_noise_report(
        cloud,
        opacity_logit_threshold=args.opacity_logit_threshold,
        scale_norm_percentile=args.scale_norm_percentile,
        voxel_size=args.voxel_size,
        isolated_min_neighbor_count=args.isolated_min_neighbor_count,
    )
    payload = json.dumps(report.to_dict(), indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
