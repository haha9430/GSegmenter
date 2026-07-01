"""Split trained identity categories into spatial instance candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.gaussian_io import load_gaussian_table
from gsegmenter.mapping.identity_instances import (
    build_identity_instance_ids,
    save_identity_instance_summary,
    write_identity_instance_highlight_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create object-instance inspection PLYs from a multi-class identity export. "
            "This clusters only high-confidence Gaussians inside selected identity classes."
        )
    )
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--identity-ids", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-filename", type=str, default="identity_instances.ply")
    parser.add_argument(
        "--include-classes",
        type=str,
        nargs="+",
        required=True,
        help="Class names or numeric class ids to split into instances, e.g. tv chair table.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="3D connected-component voxel size in exported Gaussian scene units.",
    )
    parser.add_argument(
        "--min-voxel-count",
        type=int,
        default=1,
        help="Ignore occupied voxels with fewer selected Gaussians before connectivity.",
    )
    parser.add_argument(
        "--min-gaussians",
        type=int,
        default=500,
        help="Drop connected components smaller than this Gaussian count.",
    )
    parser.add_argument(
        "--max-instances-per-class",
        type=int,
        default=0,
        help="Keep only the largest K components per class. 0 keeps all passing components.",
    )
    parser.add_argument(
        "--dim-opacity-scale",
        type=float,
        default=0.25,
        help="Opacity multiplier for Gaussians not assigned to an instance.",
    )
    return parser.parse_args()


def _load_class_names(classes_json: Path) -> list[str]:
    payload = json.loads(classes_json.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list):
        raise ValueError(f"{classes_json} does not contain a classes list.")
    max_class_id = max(int(entry["class_id"]) for entry in classes)
    names = [f"class_{index}" for index in range(max_class_id + 1)]
    for entry in classes:
        names[int(entry["class_id"])] = str(entry.get("name", f"class_{entry['class_id']}"))
    return names


def _resolve_class_ids(values: list[str], class_names: list[str]) -> set[int]:
    by_name = {name.casefold(): index for index, name in enumerate(class_names)}
    class_ids: set[int] = set()
    for raw_value in values:
        value = str(raw_value).strip()
        if value.isdigit():
            class_id = int(value)
        else:
            key = value.casefold()
            if key not in by_name:
                raise ValueError(
                    f"Unknown class {value!r}. Available classes: {', '.join(class_names)}"
                )
            class_id = by_name[key]
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"Class id {class_id} is outside available class range.")
        class_ids.add(class_id)
    return class_ids


def main() -> int:
    args = parse_args()
    table, header_properties = load_gaussian_table(args.ply_path)
    for channel in ("x", "y", "z"):
        if channel not in table.dtype.names:
            raise ValueError(f"PLY table is missing coordinate channel {channel!r}.")
    xyz = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
    identity_ids = np.load(args.identity_ids).astype(np.int32)
    probabilities = np.load(args.probabilities).astype(np.float32)
    class_names = _load_class_names(args.classes_json)
    include_class_ids = _resolve_class_ids(args.include_classes, class_names)

    instance_ids, proposals = build_identity_instance_ids(
        xyz=xyz,
        identity_ids=identity_ids,
        probabilities=probabilities,
        class_names=class_names,
        include_class_ids=include_class_ids,
        min_confidence=args.min_confidence,
        voxel_size=args.voxel_size,
        min_voxel_count=args.min_voxel_count,
        min_gaussians=args.min_gaussians,
        max_instances_per_class=args.max_instances_per_class,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename
    write_identity_instance_highlight_ply(
        output_path=output_path,
        table=table,
        header_properties=header_properties,
        instance_ids=instance_ids,
        dim_opacity_scale=args.dim_opacity_scale,
    )
    np.save(args.output_dir / "gaussian_instance_ids.npy", instance_ids)
    summary_path = args.output_dir / "identity_instances.json"
    save_identity_instance_summary(
        output_path=summary_path,
        proposals=proposals,
        parameters={
            "ply_path": str(args.ply_path),
            "identity_ids": str(args.identity_ids),
            "probabilities": str(args.probabilities),
            "classes_json": str(args.classes_json),
            "include_classes": sorted(class_names[class_id] for class_id in include_class_ids),
            "include_class_ids": sorted(int(class_id) for class_id in include_class_ids),
            "min_confidence": float(args.min_confidence),
            "voxel_size": float(args.voxel_size),
            "min_voxel_count": int(args.min_voxel_count),
            "min_gaussians": int(args.min_gaussians),
            "max_instances_per_class": int(args.max_instances_per_class),
            "dim_opacity_scale": float(args.dim_opacity_scale),
        },
    )

    print(f"Wrote instance-colored PLY to {output_path}")
    print(f"Wrote Gaussian instance ids to {args.output_dir / 'gaussian_instance_ids.npy'}")
    print(f"Wrote instance summary to {summary_path}")
    print(f"Found {len(proposals)} instance candidates")
    for proposal in proposals:
        print(
            f"  instance={proposal.instance_id} class={proposal.class_name} "
            f"rank={proposal.rank_in_class} gaussians={proposal.gaussian_count} "
            f"confidence_mean={proposal.confidence_mean:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
