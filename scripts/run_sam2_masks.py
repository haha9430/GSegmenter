"""Launch SAM 2 mask extraction from a separate Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.sam2_runner import (
    Sam2ExtractionSpec,
    run_sam2_extraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM 2 extraction using a separate Python interpreter."
    )
    parser.add_argument("--python-bin", type=str, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded directly to extract_sam2_masks.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = Sam2ExtractionSpec(
        python_bin=args.python_bin,
        script_path=Path(__file__).resolve().with_name("extract_sam2_masks.py"),
        images_dir=args.images_dir,
        output_root=args.output_root,
        checkpoint_path=args.checkpoint_path,
        model_config=args.model_config,
        limit=args.limit,
        skip_existing=args.skip_existing,
        extra_args=tuple(args.extra_args),
    )
    result = run_sam2_extraction(spec, dry_run=args.dry_run)
    if isinstance(result, list):
        print(" ".join(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
