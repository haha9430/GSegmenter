"""Prepare a NerfStudio-ready scene from an input `.mp4` capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import (
    VideoPreparationSpec,
    run_video_preparation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an input video into a NerfStudio/COLMAP dataset."
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit dataset output directory. Overrides --data-root/--scene-name.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--scene-name", type=str, default="scene01")
    parser.add_argument("--num-frames-target", type=int, default=300)
    parser.add_argument(
        "--camera-type",
        type=str,
        default="perspective",
        choices=("perspective", "fisheye", "equirectangular", "pinhole", "simple_pinhole"),
    )
    parser.add_argument(
        "--matching-method",
        type=str,
        default="sequential",
        choices=("exhaustive", "sequential", "vocab_tree"),
    )
    parser.add_argument(
        "--sfm-tool",
        type=str,
        default="colmap",
        choices=("any", "colmap", "hloc"),
    )
    parser.add_argument("--num-downscales", type=int, default=3)
    parser.add_argument("--crop-bottom", type=float, default=0.0)
    parser.add_argument(
        "--colmap-cmd",
        type=str,
        default="colmap",
        help="Executable or command used to launch COLMAP inside ns-process-data.",
    )
    parser.add_argument(
        "--ns-process-data-bin",
        type=str,
        default="ns-process-data",
        help="Executable or command used to launch NerfStudio preprocessing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", dest="gpu", action="store_true", help="Use GPU in COLMAP if available.")
    gpu_group.add_argument("--no-gpu", dest="gpu", action="store_false", help="Force CPU-only COLMAP.")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded directly to ns-process-data video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else (args.data_root / args.scene_name)
    spec = VideoPreparationSpec(
        video_path=args.video_path,
        output_dir=output_dir,
        num_frames_target=args.num_frames_target,
        camera_type=args.camera_type,
        matching_method=args.matching_method,
        sfm_tool=args.sfm_tool,
        num_downscales=args.num_downscales,
        crop_bottom=args.crop_bottom,
        colmap_cmd=args.colmap_cmd,
        gpu=args.gpu,
        verbose=args.verbose,
        ns_process_data_bin=args.ns_process_data_bin,
        extra_args=tuple(args.extra_args),
    )
    result = run_video_preparation(spec, dry_run=args.dry_run)
    if isinstance(result, list):
        print(" ".join(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
