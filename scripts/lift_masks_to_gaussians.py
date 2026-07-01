"""Lift frame-local 2D masks into sparse Gaussian vote evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud
from gsegmenter.mapping.lifting import (
    build_frame_vote_evidence,
    load_frame_manifest_from_dir,
    save_vote_evidence,
    save_vote_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift per-frame masks to Gaussian vote evidence.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="nerfstudio")
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="floor")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--use-opacity-weight", action="store_true")
    parser.add_argument("--gaussian-quality-path", type=Path, default=None)
    parser.add_argument("--valid-gaussian-mask-path", type=Path, default=None)
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--depth-fit-max-points", type=int, default=30000)
    parser.add_argument("--depth-trim-quantile", type=float, default=0.80)
    parser.add_argument("--depth-behind-margin-ratio", type=float, default=0.18)
    parser.add_argument("--depth-behind-min-margin", type=float, default=0.05)
    parser.add_argument(
        "--front-surface-only",
        action="store_true",
        help="Use z-buffer style per-pixel front-surface filtering before collecting mask votes.",
    )
    parser.add_argument("--front-surface-depth-margin", type=float, default=0.03)
    return parser.parse_args()


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

    gaussian_xyz = cloud.xyz
    opacity_weights = None
    quality_weights = None
    gaussian_valid_mask = None
    if args.max_gaussians is not None:
        gaussian_xyz = gaussian_xyz[: args.max_gaussians]
    if args.use_opacity_weight and cloud.opacities is not None:
        opacity_logits = cloud.opacities[: len(gaussian_xyz)]
        opacity_weights = 1.0 / (1.0 + np.exp(-opacity_logits))
    if args.gaussian_quality_path is not None:
        quality_weights = np.load(args.gaussian_quality_path).astype(np.float32)
        if quality_weights.shape[0] < len(gaussian_xyz):
            raise ValueError(
                f"Gaussian quality length {quality_weights.shape[0]} is smaller than "
                f"Gaussian count {len(gaussian_xyz)}"
            )
        quality_weights = quality_weights[: len(gaussian_xyz)]
    if args.valid_gaussian_mask_path is not None:
        gaussian_valid_mask = np.load(args.valid_gaussian_mask_path).astype(bool)
        if gaussian_valid_mask.shape[0] < len(gaussian_xyz):
            raise ValueError(
                f"Valid Gaussian mask length {gaussian_valid_mask.shape[0]} is smaller than "
                f"Gaussian count {len(gaussian_xyz)}"
            )
        gaussian_valid_mask = gaussian_valid_mask[: len(gaussian_xyz)]

    frames = scene.frames
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    all_evidences = []
    for frame in frames:
        frame_dir = args.masks_root / frame.file_path.stem
        manifest_path = frame_dir / "instances.json"
        if not manifest_path.exists():
            continue

        manifest = load_frame_manifest_from_dir(frame_dir)
        depth_map = None
        if args.depth_root is not None:
            depth_path = args.depth_root / frame.file_path.stem / "depth.npy"
            if not depth_path.exists():
                print(f"Skipping depth gate for {frame.file_path.name}: missing {depth_path}")
            else:
                depth_map = np.load(depth_path).astype(np.float32)
        evidences = build_frame_vote_evidence(
            gaussian_xyz=gaussian_xyz,
            intrinsics=scene.intrinsics,
            frame=frame,
            manifest=manifest,
            frame_dir=frame_dir,
            opacity_weights=opacity_weights,
            quality_weights=quality_weights,
            gaussian_valid_mask=gaussian_valid_mask,
            depth_map=depth_map,
            depth_fit_max_points=args.depth_fit_max_points,
            depth_trim_quantile=args.depth_trim_quantile,
            depth_behind_margin_ratio=args.depth_behind_margin_ratio,
            depth_behind_min_margin=args.depth_behind_min_margin,
            front_surface_only=args.front_surface_only,
            front_surface_depth_margin=args.front_surface_depth_margin,
        )
        all_evidences.extend(evidences)
        print(
            f"Processed {frame.file_path.name}: "
            f"{len(manifest.instances)} masks, {len(evidences)} vote-supporting instances"
        )

    evidence_path = args.output_root / "vote_evidence.npz"
    summary_path = args.output_root / "vote_summary.json"
    save_vote_evidence(all_evidences, evidence_path)
    save_vote_summary(all_evidences, summary_path, total_gaussians=len(gaussian_xyz))
    print(f"Saved sparse vote evidence to {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
