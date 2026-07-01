"""Launch GroundingDINO + SAM 2 mask extraction from a separate environment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grounded SAM 2 extraction using a separate Python interpreter."
    )
    parser.add_argument("--python-bin", type=str, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detector-backend", choices=("transformers", "groundingdino"), default="transformers")
    parser.add_argument("--hf-model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-config-path", type=Path, default=None)
    parser.add_argument("--grounding-checkpoint-path", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint-path", type=Path, required=True)
    parser.add_argument("--sam2-model-config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded directly to extract_grounded_sam2_masks.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        args.python_bin,
        str(Path(__file__).resolve().with_name("extract_grounded_sam2_masks.py")),
        "--images-dir",
        str(args.images_dir),
        "--output-root",
        str(args.output_root),
        "--detector-backend",
        args.detector_backend,
        "--hf-model-id",
        args.hf_model_id,
        "--sam2-checkpoint-path",
        str(args.sam2_checkpoint_path),
        "--sam2-model-config",
        args.sam2_model_config,
    ]
    if args.grounding_config_path is not None:
        command.extend(["--grounding-config-path", str(args.grounding_config_path)])
    if args.grounding_checkpoint_path is not None:
        command.extend(["--grounding-checkpoint-path", str(args.grounding_checkpoint_path)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.skip_existing:
        command.append("--skip-existing")
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command.extend(extra_args)

    if args.dry_run:
        print(" ".join(command))
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
