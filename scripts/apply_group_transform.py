"""Apply an object-local transform to one Gaussian group and export a new PLY."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.editor import apply_object_transform_about_pivot
from gsegmenter.mapping.gaussian_io import load_gaussian_table, write_gaussian_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform one grouped object inside a Gaussian PLY and write the edited scene."
    )
    parser.add_argument("--ply-path", type=Path, required=True, help="Input Gaussian PLY.")
    parser.add_argument(
        "--object-ids",
        type=Path,
        required=True,
        help="Path to gaussian_object_ids.npy aligned with the PLY row order.",
    )
    parser.add_argument("--output-path", type=Path, required=True, help="Output edited Gaussian PLY.")
    parser.add_argument("--target-object-id", type=int, required=True, help="Object id to edit.")
    parser.add_argument(
        "--translate",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="World-space translation to apply after rotation.",
    )
    parser.add_argument(
        "--rotate-axis",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("AX", "AY", "AZ"),
        help="Axis for the local rotation. Defaults to world up.",
    )
    parser.add_argument(
        "--rotate-degrees",
        type=float,
        default=0.0,
        help="Rotation angle in degrees around --rotate-axis.",
    )
    parser.add_argument(
        "--pivot-mode",
        choices=("centroid",),
        default="centroid",
        help="Pivot rule for the selected object. Currently uses the object centroid.",
    )
    return parser


def _axis_angle_to_matrix(axis_xyz: np.ndarray, degrees: float) -> np.ndarray:
    axis_xyz = np.asarray(axis_xyz, dtype=np.float32)
    norm = float(np.linalg.norm(axis_xyz))
    if norm <= 1e-8:
        raise ValueError("Rotation axis must be non-zero.")
    axis_xyz = axis_xyz / norm
    radians = np.deg2rad(np.float32(degrees))
    x, y, z = axis_xyz
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    one_minus_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float32,
    )


def main() -> int:
    args = build_parser().parse_args()
    table, header_properties = load_gaussian_table(args.ply_path)
    object_ids = np.load(args.object_ids)
    if object_ids.shape[0] != table.shape[0]:
        raise ValueError(
            f"Object id count {object_ids.shape[0]} does not match Gaussian count {table.shape[0]}"
        )

    required_columns = ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3")
    missing_columns = [name for name in required_columns if name not in table.dtype.names]
    if missing_columns:
        raise ValueError(
            "Input PLY is missing rotation/position columns required for editing: "
            + ", ".join(missing_columns)
        )

    means = torch.from_numpy(
        np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
    )
    rotations = torch.from_numpy(
        np.stack(
            [table["rot_0"], table["rot_1"], table["rot_2"], table["rot_3"]],
            axis=1,
        ).astype(np.float32)
    )
    object_ids_tensor = torch.from_numpy(object_ids.astype(np.int64))
    translation = torch.tensor(args.translate, dtype=torch.float32)
    rotation_matrix = torch.from_numpy(_axis_angle_to_matrix(np.asarray(args.rotate_axis), args.rotate_degrees))

    new_means, new_rotations = apply_object_transform_about_pivot(
        means,
        rotations,
        object_ids_tensor,
        target_id=args.target_object_id,
        translation=translation,
        rotation_matrix=rotation_matrix,
    )

    edited = table.copy()
    edited["x"] = new_means[:, 0].numpy()
    edited["y"] = new_means[:, 1].numpy()
    edited["z"] = new_means[:, 2].numpy()
    edited["rot_0"] = new_rotations[:, 0].numpy()
    edited["rot_1"] = new_rotations[:, 1].numpy()
    edited["rot_2"] = new_rotations[:, 2].numpy()
    edited["rot_3"] = new_rotations[:, 3].numpy()

    write_gaussian_table(args.output_path, edited, header_properties)
    changed_count = int(np.count_nonzero(object_ids == args.target_object_id))
    print(
        f"Wrote transformed Gaussian PLY to {args.output_path} "
        f"for object {args.target_object_id} affecting {changed_count} gaussians"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
