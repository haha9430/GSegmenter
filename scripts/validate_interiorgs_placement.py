"""Validate an InteriorGS object placement against the occupancy map."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data import load_interiorgs_scene
from gsegmenter.editor.occupancy import evaluate_interiorgs_object_placement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether a moved InteriorGS object footprint lands in valid free space."
    )
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--target-object-id", type=int, required=True)
    parser.add_argument(
        "--translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="World-space translation to test.",
    )
    parser.add_argument(
        "--rotate-degrees",
        type=float,
        default=0.0,
        help="Rotation angle in degrees around world up (Z).",
    )
    parser.add_argument("--max-occupied-fraction", type=float, default=0.05)
    parser.add_argument("--max-unknown-fraction", type=float, default=0.25)
    return parser


def _rotation_z_matrix(degrees: float) -> np.ndarray:
    radians = np.deg2rad(np.float32(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main() -> int:
    args = build_parser().parse_args()
    scene = load_interiorgs_scene(args.scene_root)
    record = None
    for candidate in scene.objects:
        if candidate.instance_id is not None and int(candidate.instance_id) == int(args.target_object_id):
            record = candidate
            break
    if record is None:
        raise KeyError(f"Could not find object id {args.target_object_id} in labels.json")

    summary = evaluate_interiorgs_object_placement(
        record,
        scene.occupancy,
        object_id=int(args.target_object_id),
        translation_xyz=np.asarray(args.translate, dtype=np.float32),
        rotation_matrix=_rotation_z_matrix(float(args.rotate_degrees)),
        max_occupied_fraction=float(args.max_occupied_fraction),
        max_unknown_fraction=float(args.max_unknown_fraction),
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
