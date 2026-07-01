"""Prepare background plus selected category labels for identity training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    save_binary_mask,
    save_frame_masks_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert grounded masks into one multi-class identity dataset with "
            "an explicit full-frame background class."
        )
    )
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--category",
        action="append",
        nargs=3,
        metavar=("NAME", "ID", "LABEL_REGEX"),
        required=True,
        help="Category name, integer class id, and regex used to match grounded labels.",
    )
    parser.add_argument("--background-name", type=str, default="background")
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--background-score", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--max-instances-per-category-per-frame", type=int, default=None)
    parser.add_argument("--include-empty-background-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"frame_index", "image_path", "image_size", "instances"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
    return payload


def _discover_source_manifests(masks_root: Path) -> tuple[Path, ...]:
    manifests = tuple(sorted(Path(masks_root).glob("*/instances.json")))
    if not manifests:
        raise FileNotFoundError(f"No per-frame instances.json files were found under {masks_root}")
    return manifests


def _candidate_score(instance: dict) -> float:
    detection_score = instance.get("detection_score")
    if detection_score is not None:
        return float(detection_score)
    return float(instance.get("score", 0.0))


def _parse_categories(raw_categories: list[list[str]], background_id: int) -> list[tuple[str, int, re.Pattern[str]]]:
    categories: list[tuple[str, int, re.Pattern[str]]] = []
    seen_ids = {int(background_id)}
    seen_names = {"background"}
    for name, class_id_raw, pattern_raw in raw_categories:
        class_id = int(class_id_raw)
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("Category names must not be empty.")
        if normalized_name.casefold() in seen_names:
            raise ValueError(f"Duplicate category name: {normalized_name}")
        if class_id in seen_ids:
            raise ValueError(f"Duplicate category id: {class_id}")
        seen_names.add(normalized_name.casefold())
        seen_ids.add(class_id)
        categories.append((normalized_name, class_id, re.compile(pattern_raw, flags=re.IGNORECASE)))
    return categories


def _select_instances(
    payload: dict,
    categories: list[tuple[str, int, re.Pattern[str]]],
    *,
    min_score: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_instances_per_category_per_frame: int | None,
) -> tuple[list[MaskInstanceRecord], list[dict[str, object]]]:
    width, height = (int(value) for value in payload["image_size"])
    image_area = max(width * height, 1)
    selected_records: list[MaskInstanceRecord] = []
    selected_summary: list[dict[str, object]] = []

    for category_name, class_id, pattern in categories:
        candidates = []
        for instance in payload["instances"]:
            label = str(instance.get("label", ""))
            if pattern.search(label) is None:
                continue
            score = _candidate_score(instance)
            area_ratio = float(instance.get("area", 0)) / image_area
            if score < min_score:
                continue
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            enriched = dict(instance)
            enriched["_candidate_score"] = score
            enriched["_area_ratio"] = area_ratio
            enriched["_category_name"] = category_name
            enriched["_class_id"] = class_id
            candidates.append(enriched)
        candidates.sort(
            key=lambda instance: (
                float(instance["_candidate_score"]),
                float(instance["_area_ratio"]),
            ),
            reverse=True,
        )
        if max_instances_per_category_per_frame is not None:
            candidates = candidates[: int(max_instances_per_category_per_frame)]
        for instance in candidates:
            selected_records.append(
                MaskInstanceRecord(
                    instance_id=int(class_id),
                    bbox_xyxy=tuple(int(value) for value in instance["bbox_xyxy"]),
                    score=float(instance["_candidate_score"]),
                    area=int(instance.get("area", 0)),
                    mask_path=str(instance["mask_path"]),
                )
            )
            selected_summary.append(
                {
                    "category": category_name,
                    "class_id": int(class_id),
                    "source_instance_id": int(instance["instance_id"]),
                    "label": str(instance.get("label", "")),
                    "score": float(instance.get("score", 0.0)),
                    "candidate_score": float(instance["_candidate_score"]),
                    "area": int(instance.get("area", 0)),
                    "area_ratio": float(instance["_area_ratio"]),
                    "bbox_xyxy": [int(value) for value in instance["bbox_xyxy"]],
                    "source_mask_path": str(instance["mask_path"]),
                }
            )
    return selected_records, selected_summary


def _convert_frame(
    manifest_path: Path,
    output_frame_dir: Path,
    selected_records: list[MaskInstanceRecord],
    *,
    background_id: int,
    background_score: float,
) -> tuple[int, int]:
    payload = _load_manifest(manifest_path)
    width, height = (int(value) for value in payload["image_size"])
    output_frame_dir.mkdir(parents=True, exist_ok=True)

    output_records = [
        MaskInstanceRecord(
            instance_id=int(background_id),
            bbox_xyxy=(0, 0, width - 1, height - 1),
            score=float(background_score),
            area=int(width * height),
            mask_path="mask_background.png",
        )
    ]
    save_binary_mask(np.ones((height, width), dtype=bool), output_frame_dir / "mask_background.png")

    copied_count = 0
    for output_index, record in enumerate(selected_records):
        source_mask = manifest_path.parent / record.mask_path
        if not source_mask.exists():
            raise FileNotFoundError(f"Mask file not found: {source_mask}")
        output_mask = output_frame_dir / f"mask_{output_index:04d}.png"
        shutil.copy2(source_mask, output_mask)
        copied_count += 1
        output_records.append(
            MaskInstanceRecord(
                instance_id=record.instance_id,
                bbox_xyxy=record.bbox_xyxy,
                score=record.score,
                area=record.area,
                mask_path=output_mask.name,
            )
        )

    manifest = FrameMasksManifest(
        frame_index=int(payload["frame_index"]),
        image_path=str(payload["image_path"]),
        image_size=(width, height),
        instances=tuple(output_records),
    )
    save_frame_masks_manifest(manifest, output_frame_dir / "instances.json")
    return len(output_records), copied_count


def main() -> int:
    args = parse_args()
    if args.background_id < 0:
        raise ValueError("--background-id must be non-negative.")
    if args.min_area_ratio < 0.0 or args.max_area_ratio > 1.0 or args.min_area_ratio > args.max_area_ratio:
        raise ValueError("Area ratio bounds must satisfy 0 <= min <= max <= 1.")
    if args.max_instances_per_category_per_frame is not None and args.max_instances_per_category_per_frame <= 0:
        raise ValueError("--max-instances-per-category-per-frame must be positive when provided.")

    categories = _parse_categories(args.category, int(args.background_id))
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_paths = _discover_source_manifests(args.masks_root)
    frame_summaries = []
    total_instances = 0
    total_category_instances = 0
    output_frame_index = 0
    category_counts = {name: 0 for name, _, _ in categories}
    for manifest_path in manifest_paths:
        payload = _load_manifest(manifest_path)
        selected_records, selected_summary = _select_instances(
            payload,
            categories,
            min_score=args.min_score,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            max_instances_per_category_per_frame=args.max_instances_per_category_per_frame,
        )
        if not selected_records and not args.include_empty_background_frames:
            continue
        output_frame_dir = args.output_root / f"frame_{output_frame_index:05d}"
        instance_count, copied_count = _convert_frame(
            manifest_path,
            output_frame_dir,
            selected_records,
            background_id=int(args.background_id),
            background_score=float(args.background_score),
        )
        output_frame_index += 1
        total_instances += instance_count
        total_category_instances += copied_count
        for item in selected_summary:
            category_counts[str(item["category"])] += 1
        frame_summaries.append(
            {
                "source_manifest": str(manifest_path),
                "source_frame": manifest_path.parent.name,
                "output_frame": output_frame_dir.name,
                "instance_count": instance_count,
                "category_instance_count": copied_count,
                "selected": selected_summary,
            }
        )

    if total_category_instances == 0:
        raise ValueError("No category masks matched the requested filters.")

    classes = [{"raw_key": args.background_name, "global_id": int(args.background_id)}]
    classes.extend(
        {"raw_key": name, "global_id": int(class_id)}
        for name, class_id, _ in sorted(categories, key=lambda item: item[1])
    )
    payload = {
        "background_name": args.background_name,
        "background_id": int(args.background_id),
        "min_score": float(args.min_score),
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "max_instances_per_category_per_frame": args.max_instances_per_category_per_frame,
        "class_count": len(classes),
        "classes": classes,
        "category_counts": category_counts,
        "frame_count": len(frame_summaries),
        "total_instances": total_instances,
        "total_category_instances": total_category_instances,
        "frames": frame_summaries,
    }
    (args.output_root / "identity_mask_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(frame_summaries)} frames with {total_category_instances} category masks "
        f"and {len(classes)} classes to {args.output_root}"
    )
    for class_entry in classes:
        print(f"  id={class_entry['global_id']} key={class_entry['raw_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
