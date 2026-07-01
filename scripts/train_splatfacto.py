"""CLI wrapper for NerfStudio Splatfacto training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.conf import AppConfig
from gsegmenter.training.splatfacto import (
    SplatfactoTrainingSpec,
    run_splatfacto_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NerfStudio Splatfacto training.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Explicit NerfStudio dataset root. Overrides --data-root/--scene-name.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--scene-name", type=str, default="scene01")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--data-parser",
        type=str,
        default=None,
        help="Optional NerfStudio data parser subcommand such as 'colmap'.",
    )
    parser.add_argument("--num-iterations", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument(
        "--ns-train-bin",
        type=str,
        default="ns-train",
        help="Executable or command used to launch NerfStudio.",
    )
    parser.add_argument("--dry-run", action="store_true")
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        action="store_true",
        help="Enable mixed precision training.",
    )
    precision_group.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable mixed precision training for debugging or compatibility testing.",
    )
    parser.set_defaults(mixed_precision=False)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded directly to ns-train.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig()
    config.dataset.data_root = args.data_root
    config.dataset.scene_name = args.scene_name
    config.training.output_root = args.output_root
    config.training.num_iterations = args.num_iterations
    config.training.seed = args.seed
    config.training.eval_interval = args.eval_interval
    config.training.save_interval = args.save_interval
    config.training.mixed_precision = args.mixed_precision

    data_path = args.data_path if args.data_path is not None else config.dataset.scene_root
    spec = SplatfactoTrainingSpec(
        data_path=data_path,
        output_dir=config.training.output_root / config.dataset.scene_name,
        num_iterations=config.training.num_iterations,
        seed=config.training.seed,
        eval_interval=config.training.eval_interval,
        save_interval=config.training.save_interval,
        mixed_precision=config.training.mixed_precision,
        ns_train_bin=args.ns_train_bin,
        data_parser=args.data_parser,
        extra_args=tuple(args.extra_args),
    )
    result = run_splatfacto_training(spec, dry_run=args.dry_run)
    if isinstance(result, list):
        print(" ".join(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
