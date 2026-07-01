"""CLI entrypoint for identity-aware Splatfacto training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from gsegmenter.conf import AppConfig
from gsegmenter.training.identity_runner import (
    IdentitySplatfactoTrainingSpec,
    run_identity_splatfacto_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run identity-aware NerfStudio Splatfacto training.")
    parser.add_argument("--data-path", type=Path, default=None, help="Explicit NerfStudio dataset root.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--scene-name", type=str, default="scene01")
    parser.add_argument("--masks-root", type=Path, default=None, help="Root containing frame_*/instances.json masks.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--scene-format", choices=("auto", "nerfstudio", "colmap"), default="auto")
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="floor")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--num-iterations", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--cull-alpha-thresh", type=float, default=0.1)
    parser.add_argument("--cull-scale-thresh", type=float, default=0.5)
    parser.add_argument("--reset-alpha-every", type=int, default=30)
    parser.add_argument("--densify-grad-thresh", type=float, default=0.0008)
    parser.add_argument("--max-gauss-ratio", type=float, default=10.0)
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Spherical harmonics degree. Must match the loaded baseline checkpoint for identity-only runs.",
    )
    parser.add_argument("--identity-min-mask-score", type=float, default=0.5)
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="Freeze loaded Gaussian geometry/RGB and train only identity embeddings/classifier.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Optional baseline Splatfacto checkpoint used to initialize the identity-aware model.",
    )
    parser.add_argument("--dry-run", action="store_true")
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--quit-on-train-completion", dest="quit_on_train_completion", action="store_true")
    viewer_group.add_argument("--keep-viewer-alive", dest="quit_on_train_completion", action="store_false")
    parser.set_defaults(quit_on_train_completion=True)
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    precision_group.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=False)
    regularization_group = parser.add_mutually_exclusive_group()
    regularization_group.add_argument(
        "--use-scale-regularization",
        dest="use_scale_regularization",
        action="store_true",
    )
    regularization_group.add_argument(
        "--no-scale-regularization",
        dest="use_scale_regularization",
        action="store_false",
    )
    parser.set_defaults(use_scale_regularization=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig()
    config.dataset.data_root = args.data_root
    config.dataset.scene_name = args.scene_name
    config.training.output_root = args.output_root

    data_path = args.data_path if args.data_path is not None else config.dataset.scene_root
    masks_root = args.masks_root if args.masks_root is not None else args.output_root / args.scene_name / "masks"

    spec = IdentitySplatfactoTrainingSpec(
        data_path=data_path,
        masks_root=masks_root,
        output_dir=args.output_root / args.scene_name,
        scene_format=args.scene_format,
        downscale_factor=args.downscale_factor,
        downscale_rounding_mode=args.downscale_rounding_mode,
        images_path=args.images_path,
        colmap_path=args.colmap_path,
        num_iterations=args.num_iterations,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        quit_on_train_completion=args.quit_on_train_completion,
        cull_alpha_thresh=args.cull_alpha_thresh,
        cull_scale_thresh=args.cull_scale_thresh,
        reset_alpha_every=args.reset_alpha_every,
        densify_grad_thresh=args.densify_grad_thresh,
        use_scale_regularization=args.use_scale_regularization,
        max_gauss_ratio=args.max_gauss_ratio,
        sh_degree=args.sh_degree,
        identity_min_mask_score=args.identity_min_mask_score,
        identity_only=args.identity_only,
        load_checkpoint=args.load_checkpoint,
    )
    result = run_identity_splatfacto_training(spec, dry_run=args.dry_run)
    if args.dry_run:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
