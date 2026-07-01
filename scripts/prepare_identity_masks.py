"""Convert extracted 2D masks into Gaussian Grouping style identity labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import infer_label_family
from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    save_frame_masks_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-frame Grounded/SAM2 mask folders into the frame_*/instances.json "
            "layout consumed by identity-aware Splatfacto training."
        )
    )
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--id-mode",
        choices=("label-family", "label", "local-instance"),
        default="label-family",
        help=(
            "How to create scene-global IDs. label-family is stable but semantic-category level; "
            "local-instance preserves per-frame IDs and is not multi-view consistent."
        ),
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--drop-null-family", action="store_true", default=True)
    parser.add_argument("--keep-null-family", dest="drop_null_family", action="store_false")
    parser.add_argument("--max-instances-per-frame", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"frame_index", "image_path", "image_size", "instances"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    return payload


def _instance_key(instance: dict, mode: str) -> str | int:
    if mode == "local-instance":
        return int(instance["instance_id"])
    label = str(instance.get("label", "unknown")).strip().casefold()
    if mode == "label":
        return label or "unknown"
    return infer_label_family(label) or "null"


def _discover_source_manifests(masks_root: Path) -> tuple[Path, ...]:
    manifests = tuple(sorted(Path(masks_root).glob("*/instances.json")))
    if not manifests:
        raise FileNotFoundError(f"No per-frame instances.json files were found under {masks_root}")
    return manifests


def _build_global_id_map(
    manifest_paths: tuple[Path, ...],
    *,
    mode: str,
    min_score: float,
    drop_null_family: bool,
) -> dict[str | int, int]:
    keys: set[str | int] = set()
    for manifest_path in manifest_paths:
        payload = _load_manifest(manifest_path)
        for instance in payload["instances"]:
            if float(instance.get("score", 0.0)) < min_score:
                continue
            key = _instance_key(instance, mode)
            if mode == "label-family" and drop_null_family and key == "null":
                continue
            keys.add(key)

    if mode == "local-instance":
        ordered_keys = sorted(keys, key=lambda value: int(value))
    else:
        ordered_keys = sorted(keys, key=lambda value: str(value))
    return {key: index for index, key in enumerate(ordered_keys)}


def _convert_one_frame(
    manifest_path: Path,
    output_frame_dir: Path,
    *,
    global_id_by_key: dict[str | int, int],
    mode: str,
    min_score: float,
    drop_null_family: bool,
    max_instances_per_frame: int | None,
) -> tuple[int, int]:
    payload = _load_manifest(manifest_path)
    instances = [
        instance
        for instance in payload["instances"]
        if float(instance.get("score", 0.0)) >= min_score
    ]
    instances.sort(key=lambda instance: float(instance.get("score", 0.0)), reverse=True)
    if max_instances_per_frame is not None:
        instances = instances[: int(max_instances_per_frame)]

    output_frame_dir.mkdir(parents=True, exist_ok=True)
    converted_instances: list[MaskInstanceRecord] = []
    copied_count = 0
    for instance in instances:
        key = _instance_key(instance, mode)
        if mode == "label-family" and drop_null_family and key == "null":
            continue
        if key not in global_id_by_key:
            continue
        source_mask = manifest_path.parent / str(instance["mask_path"])
        if not source_mask.exists():
            raise FileNotFoundError(f"Mask file not found: {source_mask}")
        output_mask = output_frame_dir / source_mask.name
        shutil.copy2(source_mask, output_mask)
        copied_count += 1
        converted_instances.append(
            MaskInstanceRecord(
                instance_id=int(global_id_by_key[key]),
                bbox_xyxy=tuple(int(value) for value in instance["bbox_xyxy"]),
                score=float(instance.get("score", 1.0)),
                area=int(instance.get("area", 0)),
                mask_path=output_mask.name,
            )
        )

    manifest = FrameMasksManifest(
        frame_index=int(payload["frame_index"]),
        image_path=str(payload["image_path"]),
        image_size=tuple(int(value) for value in payload["image_size"]),
        instances=tuple(converted_instances),
    )
    save_frame_masks_manifest(manifest, output_frame_dir / "instances.json")
    return len(converted_instances), copied_count


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_paths = _discover_source_manifests(args.masks_root)
    global_id_by_key = _build_global_id_map(
        manifest_paths,
        mode=args.id_mode,
        min_score=args.min_score,
        drop_null_family=args.drop_null_family,
    )
    if not global_id_by_key:
        raise ValueError("No identity classes remained after filtering masks.")

    frame_summaries = []
    total_instances = 0
    for output_index, manifest_path in enumerate(manifest_paths):
        output_frame_dir = args.output_root / f"frame_{output_index:05d}"
        instance_count, copied_count = _convert_one_frame(
            manifest_path,
            output_frame_dir,
            global_id_by_key=global_id_by_key,
            mode=args.id_mode,
            min_score=args.min_score,
            drop_null_family=args.drop_null_family,
            max_instances_per_frame=args.max_instances_per_frame,
        )
        total_instances += instance_count
        frame_summaries.append(
            {
                "source_manifest": str(manifest_path),
                "output_frame": output_frame_dir.name,
                "instance_count": instance_count,
                "copied_masks": copied_count,
            }
        )

    mapping_payload = {
        "id_mode": args.id_mode,
        "min_score": float(args.min_score),
        "drop_null_family": bool(args.drop_null_family),
        "class_count": len(global_id_by_key),
        "classes": [
            {"raw_key": str(key), "global_id": int(value)}
            for key, value in sorted(global_id_by_key.items(), key=lambda item: item[1])
        ],
        "frame_count": len(frame_summaries),
        "total_instances": total_instances,
        "frames": frame_summaries,
    }
    (args.output_root / "identity_mask_manifest.json").write_text(
        json.dumps(mapping_payload, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(frame_summaries)} identity mask frames with "
        f"{total_instances} instances and {len(global_id_by_key)} classes to {args.output_root}"
    )
    for entry in mapping_payload["classes"]:
        print(f"  id={entry['global_id']} key={entry['raw_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
