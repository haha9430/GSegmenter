"""Convert InteriorGS `3dgs_compressed.ply` into standard PLY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import InteriorGSConvertSpec, run_interiorgs_convert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert InteriorGS compressed PLY into standard PLY.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--splat-transform-bin",
        type=str,
        default="splat-transform",
        help="Executable or command used to launch PlayCanvas splat-transform.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded directly to splat-transform.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = InteriorGSConvertSpec(
        input_path=args.input_path,
        output_path=args.output_path,
        splat_transform_bin=args.splat_transform_bin,
        overwrite=args.overwrite,
        extra_args=tuple(args.extra_args),
    )
    result = run_interiorgs_convert(spec, dry_run=args.dry_run)
    if isinstance(result, list):
        print(" ".join(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
