"""Build identity supervision for one manually targeted object category."""

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
            "Select masks matching one object label pattern and rewrite them as one "
            "scene-global identity id for object-level Gaussian Grouping experiments."
        )
    )
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label-regex", type=str, required=True)
    parser.add_argument("--object-name", type=str, default="object_00")
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument(
        "--add-background-class",
        action="store_true",
        help=(
            "Add a full-frame background mask so single-object training learns "
            "background vs object instead of a degenerate one-class softmax."
        ),
    )
    parser.add_argument("--background-name", type=str, default="background")
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument(
        "--background-score",
        type=float,
        default=0.0,
        help=(
            "Score assigned to the synthetic background mask. Use "
            "--identity-min-mask-score 0.0 when training with the default."
        ),
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument(
        "--source-frame-regex",
        type=str,
        default=None,
        help="Optional regex applied to the source frame folder name before selecting masks.",
    )
    parser.add_argument("--bbox-center-x-min", type=float, default=0.0)
    parser.add_argument("--bbox-center-x-max", type=float, default=1.0)
    parser.add_argument("--bbox-center-y-min", type=float, default=0.0)
    parser.add_argument("--bbox-center-y-max", type=float, default=1.0)
    parser.add_argument("--single-best-per-frame", action="store_true", default=True)
    parser.add_argument("--keep-all-per-frame", dest="single_best_per_frame", action="store_false")
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


def _select_candidates(
    payload: dict,
    label_pattern: re.Pattern[str],
    *,
    min_score: float,
    min_area_ratio: float,
    max_area_ratio: float,
    bbox_center_x_min: float,
    bbox_center_x_max: float,
    bbox_center_y_min: float,
    bbox_center_y_max: float,
    single_best_per_frame: bool,
) -> list[dict]:
    width, height = (int(value) for value in payload["image_size"])
    image_area = max(width * height, 1)
    candidates: list[dict] = []
    for instance in payload["instances"]:
        label = str(instance.get("label", ""))
        if label_pattern.search(label) is None:
            continue
        score = _candidate_score(instance)
        area_ratio = float(instance.get("area", 0)) / image_area
        if score < min_score:
            continue
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        x0, y0, x1, y1 = (float(value) for value in instance["bbox_xyxy"])
        center_x = ((x0 + x1) * 0.5) / max(width, 1)
        center_y = ((y0 + y1) * 0.5) / max(height, 1)
        if center_x < bbox_center_x_min or center_x > bbox_center_x_max:
            continue
        if center_y < bbox_center_y_min or center_y > bbox_center_y_max:
            continue
        enriched = dict(instance)
        enriched["_candidate_score"] = score
        enriched["_area_ratio"] = area_ratio
        enriched["_bbox_center"] = (center_x, center_y)
        candidates.append(enriched)

    candidates.sort(
        key=lambda instance: (
            float(instance["_candidate_score"]),
            float(instance["_area_ratio"]),
        ),
        reverse=True,
    )
    if single_best_per_frame and candidates:
        return [candidates[0]]
    return candidates


def _convert_frame(
    manifest_path: Path,
    output_frame_dir: Path,
    candidates: list[dict],
    *,
    object_id: int,
    add_background_class: bool,
    background_id: int,
    background_score: float,
) -> int:
    payload = _load_manifest(manifest_path)
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    converted: list[MaskInstanceRecord] = []
    width, height = (int(value) for value in payload["image_size"])
    if add_background_class:
        background_mask_path = output_frame_dir / "mask_background.png"
        save_binary_mask(np.ones((height, width), dtype=bool), background_mask_path)
        converted.append(
            MaskInstanceRecord(
                instance_id=int(background_id),
                bbox_xyxy=(0, 0, width - 1, height - 1),
                score=float(background_score),
                area=int(width * height),
                mask_path=background_mask_path.name,
            )
        )
    for output_index, instance in enumerate(candidates):
        source_mask = manifest_path.parent / str(instance["mask_path"])
        if not source_mask.exists():
            raise FileNotFoundError(f"Mask file not found: {source_mask}")
        output_mask = output_frame_dir / f"mask_{output_index:04d}.png"
        shutil.copy2(source_mask, output_mask)
        converted.append(
            MaskInstanceRecord(
                instance_id=int(object_id),
                bbox_xyxy=tuple(int(value) for value in instance["bbox_xyxy"]),
                score=float(instance.get("score", instance["_candidate_score"])),
                area=int(instance.get("area", 0)),
                mask_path=output_mask.name,
            )
        )

    manifest = FrameMasksManifest(
        frame_index=int(payload["frame_index"]),
        image_path=str(payload["image_path"]),
        image_size=tuple(int(value) for value in payload["image_size"]),
        instances=tuple(converted),
    )
    save_frame_masks_manifest(manifest, output_frame_dir / "instances.json")
    return len(converted)


def main() -> int:
    args = parse_args()
    if args.object_id < 0:
        raise ValueError("--object-id must be non-negative.")
    if args.background_id < 0:
        raise ValueError("--background-id must be non-negative.")
    if args.add_background_class and args.background_id == args.object_id:
        raise ValueError("--background-id and --object-id must differ when adding a background class.")
    if args.min_area_ratio < 0.0 or args.max_area_ratio > 1.0 or args.min_area_ratio > args.max_area_ratio:
        raise ValueError("Area ratio bounds must satisfy 0 <= min <= max <= 1.")
    bbox_bounds = (
        args.bbox_center_x_min,
        args.bbox_center_x_max,
        args.bbox_center_y_min,
        args.bbox_center_y_max,
    )
    if any(value < 0.0 or value > 1.0 for value in bbox_bounds):
        raise ValueError("Bounding-box center filters must be normalized to [0, 1].")
    if args.bbox_center_x_min > args.bbox_center_x_max or args.bbox_center_y_min > args.bbox_center_y_max:
        raise ValueError("Bounding-box center min values must not exceed max values.")
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    label_pattern = re.compile(args.label_regex, flags=re.IGNORECASE)
    source_frame_pattern = (
        re.compile(args.source_frame_regex, flags=re.IGNORECASE) if args.source_frame_regex is not None else None
    )
    manifest_paths = _discover_source_manifests(args.masks_root)
    frame_summaries = []
    selected_count = 0
    selected_frame_index = 0
    for manifest_path in manifest_paths:
        if source_frame_pattern is not None and source_frame_pattern.search(manifest_path.parent.name) is None:
            continue
        payload = _load_manifest(manifest_path)
        candidates = _select_candidates(
            payload,
            label_pattern,
            min_score=args.min_score,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            bbox_center_x_min=args.bbox_center_x_min,
            bbox_center_x_max=args.bbox_center_x_max,
            bbox_center_y_min=args.bbox_center_y_min,
            bbox_center_y_max=args.bbox_center_y_max,
            single_best_per_frame=args.single_best_per_frame,
        )
        if not candidates:
            continue
        output_frame_dir = args.output_root / f"frame_{selected_frame_index:05d}"
        converted_count = _convert_frame(
            manifest_path,
            output_frame_dir,
            candidates,
            object_id=args.object_id,
            add_background_class=args.add_background_class,
            background_id=args.background_id,
            background_score=args.background_score,
        )
        selected_count += converted_count
        selected_frame_index += 1
        frame_summaries.append(
            {
                "source_manifest": str(manifest_path),
                "source_frame": manifest_path.parent.name,
                "output_frame": output_frame_dir.name,
                "selected_count": converted_count,
                "selected": [
                    {
                        "source_instance_id": int(instance["instance_id"]),
                        "label": str(instance.get("label", "")),
                        "score": float(instance.get("score", 0.0)),
                        "candidate_score": float(instance["_candidate_score"]),
                        "area": int(instance.get("area", 0)),
                        "area_ratio": float(instance["_area_ratio"]),
                        "bbox_center": [float(value) for value in instance["_bbox_center"]],
                        "bbox_xyxy": [int(value) for value in instance["bbox_xyxy"]],
                        "source_mask_path": str(manifest_path.parent / str(instance["mask_path"])),
                    }
                    for instance in candidates
                ],
            }
        )

    if selected_count == 0:
        raise ValueError("No masks matched the requested single-object filters.")

    payload = {
        "object_name": args.object_name,
        "object_id": int(args.object_id),
        "background_name": args.background_name if args.add_background_class else None,
        "background_id": int(args.background_id) if args.add_background_class else None,
        "label_regex": args.label_regex,
        "min_score": float(args.min_score),
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "source_frame_regex": args.source_frame_regex,
        "bbox_center_filter": {
            "x_min": float(args.bbox_center_x_min),
            "x_max": float(args.bbox_center_x_max),
            "y_min": float(args.bbox_center_y_min),
            "y_max": float(args.bbox_center_y_max),
        },
        "single_best_per_frame": bool(args.single_best_per_frame),
        "class_count": 2 if args.add_background_class else 1,
        "classes": (
            [
                {"raw_key": args.background_name, "global_id": int(args.background_id)},
                {"raw_key": args.object_name, "global_id": int(args.object_id)},
            ]
            if args.add_background_class
            else [{"raw_key": args.object_name, "global_id": int(args.object_id)}]
        ),
        "frame_count": len(frame_summaries),
        "total_instances": selected_count,
        "frames": frame_summaries,
    }
    (args.output_root / "identity_mask_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {selected_count} masks across {len(frame_summaries)} frames "
        f"for object {args.object_name!r} to {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
