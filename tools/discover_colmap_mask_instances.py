"""Build second-pass identity masks from Grounded-SAM masks using COLMAP tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.colmap_tracks import read_colmap_images_binary
from gsegmenter.mapping.category_discovery import default_category_specs, match_category
from gsegmenter.mapping.colmap_mask_discovery import (
    TrackMaskEvidence,
    assign_track_instance_groups,
    collect_mask_track_ids,
)
from gsegmenter.segmentation.mask_io import FrameMasksManifest, MaskInstanceRecord, save_binary_mask, save_frame_masks_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PLY-free first-pass discovery: associate Grounded-SAM dense masks "
            "through shared COLMAP sparse point tracks and write identity masks."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--grounded-masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--colmap-images-bin", type=Path, default=Path("colmap/sparse/0/images.bin"))
    parser.add_argument("--colmap-image-width", type=int, required=True)
    parser.add_argument("--colmap-image-height", type=int, required=True)
    parser.add_argument("--include-categories", nargs="*", default=("tv", "chair", "table", "sofa", "storage"))
    parser.add_argument("--min-mask-score", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-track-points", type=int, default=8)
    parser.add_argument("--min-shared-points", type=int, default=4)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.08)
    parser.add_argument("--min-group-masks", type=int, default=3)
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--background-score", type=float, default=0.0)
    parser.add_argument("--include-empty-background-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest_payload(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"frame_index", "image_path", "image_size", "instances"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{manifest_path} is missing required keys: {sorted(missing)}")
    return payload


def _candidate_score(instance: dict) -> float:
    if instance.get("detection_score") is not None:
        return float(instance["detection_score"])
    return float(instance.get("score", 0.0))


def _group_class_name(group_category: str, rank_in_category: int) -> str:
    return f"{group_category}_{rank_in_category + 1:02d}"


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    colmap_images_bin = args.colmap_images_bin
    if not colmap_images_bin.is_absolute():
        colmap_images_bin = args.dataset_root / colmap_images_bin
    tracks_by_stem = read_colmap_images_binary(colmap_images_bin)
    all_specs = default_category_specs()
    requested = {str(value).casefold() for value in args.include_categories}
    specs = [spec for spec in all_specs if spec.name.casefold() in requested]
    if not specs:
        raise ValueError("No categories selected.")

    evidences: list[TrackMaskEvidence] = []
    selected_mask_count = 0
    category_mask_counts = {spec.name: 0 for spec in specs}
    manifest_payloads: dict[str, dict] = {}
    manifest_paths = tuple(sorted(args.grounded_masks_root.glob("*/instances.json")))
    for manifest_path in manifest_paths:
        frame_stem = manifest_path.parent.name
        tracks = tracks_by_stem.get(frame_stem)
        if tracks is None:
            continue
        payload = _load_manifest_payload(manifest_path)
        manifest_payloads[frame_stem] = payload
        width, height = (int(value) for value in payload["image_size"])
        image_area = max(width * height, 1)
        for instance in payload["instances"]:
            label = str(instance.get("label", ""))
            category_index = match_category(label, specs)
            if category_index is None:
                continue
            score = _candidate_score(instance)
            area = int(instance.get("area", 0))
            area_ratio = area / image_area
            if score < args.min_mask_score or area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
                continue
            point_ids = collect_mask_track_ids(
                track_xy=tracks.xy,
                point3d_ids=tracks.point3d_ids,
                mask_path=manifest_path.parent / str(instance["mask_path"]),
                colmap_image_size=(args.colmap_image_width, args.colmap_image_height),
                mask_image_size=(width, height),
            )
            selected_mask_count += 1
            if point_ids.size < int(args.min_track_points):
                continue
            category = specs[category_index].name
            category_mask_counts[category] += 1
            evidences.append(
                TrackMaskEvidence(
                    local_index=len(evidences),
                    frame_stem=frame_stem,
                    source_instance_id=int(instance["instance_id"]),
                    category=category,
                    label=label,
                    mask_path=str(instance["mask_path"]),
                    score=score,
                    area=area,
                    bbox_xyxy=tuple(int(value) for value in instance["bbox_xyxy"]),
                    point3d_ids=point_ids,
                )
            )

    group_ids, groups = assign_track_instance_groups(
        evidences,
        min_shared_points=args.min_shared_points,
        min_overlap_ratio=args.min_overlap_ratio,
        min_group_masks=args.min_group_masks,
    )
    category_rank: dict[str, int] = {}
    class_by_group_id = {}
    class_names = {}
    for group in groups:
        rank = category_rank.get(group.category, 0)
        category_rank[group.category] = rank + 1
        class_by_group_id[group.group_id] = len(class_by_group_id) + 1
        class_names[group.group_id] = _group_class_name(group.category, rank)

    evidence_by_frame: dict[str, list[tuple[TrackMaskEvidence, int]]] = {}
    for evidence, group_id in zip(evidences, group_ids, strict=True):
        if group_id < 0:
            continue
        evidence_by_frame.setdefault(evidence.frame_stem, []).append((evidence, int(group_id)))

    frame_summaries = []
    assigned_mask_count = 0
    class_mask_counts = {name: 0 for name in class_names.values()}
    for output_frame_index, manifest_path in enumerate(manifest_paths):
        frame_stem = manifest_path.parent.name
        payload = manifest_payloads.get(frame_stem)
        if payload is None:
            payload = _load_manifest_payload(manifest_path)
        width, height = (int(value) for value in payload["image_size"])
        assignments = evidence_by_frame.get(frame_stem, [])
        if not assignments and not args.include_empty_background_frames:
            continue
        output_frame_dir = args.output_root / f"frame_{output_frame_index:05d}"
        records = [
            MaskInstanceRecord(
                instance_id=int(args.background_id),
                bbox_xyxy=(0, 0, width - 1, height - 1),
                score=float(args.background_score),
                area=int(width * height),
                mask_path="mask_background.png",
            )
        ]
        save_binary_mask(np.ones((height, width), dtype=bool), output_frame_dir / "mask_background.png")
        assigned = []
        for mask_index, (evidence, group_id) in enumerate(assignments):
            mask_name = f"mask_{mask_index:04d}.png"
            shutil.copy2(manifest_path.parent / evidence.mask_path, output_frame_dir / mask_name)
            class_id = int(class_by_group_id[group_id])
            records.append(
                MaskInstanceRecord(
                    instance_id=class_id,
                    bbox_xyxy=evidence.bbox_xyxy,
                    score=float(evidence.score),
                    area=int(evidence.area),
                    mask_path=mask_name,
                )
            )
            assigned_mask_count += 1
            class_mask_counts[class_names[group_id]] += 1
            assigned.append(
                {
                    "source_frame": evidence.frame_stem,
                    "source_instance_id": evidence.source_instance_id,
                    "source_label": evidence.label,
                    "group_id": int(group_id),
                    "class_id": class_id,
                    "class_name": class_names[group_id],
                    "track_support": evidence.support_size,
                    "area": evidence.area,
                    "bbox_xyxy": list(evidence.bbox_xyxy),
                }
            )
        save_frame_masks_manifest(
            FrameMasksManifest(
                frame_index=len(frame_summaries),
                image_path=str(payload["image_path"]),
                image_size=(width, height),
                instances=tuple(records),
            ),
            output_frame_dir / "instances.json",
        )
        frame_summaries.append(
            {
                "output_frame": output_frame_dir.name,
                "source_frame": frame_stem,
                "assigned_mask_count": len(assignments),
                "assigned": assigned,
            }
        )

    classes = [{"raw_key": "background", "global_id": int(args.background_id)}]
    for group in groups:
        classes.append(
            {
                "raw_key": class_names[group.group_id],
                "global_id": int(class_by_group_id[group.group_id]),
                "group_id": int(group.group_id),
                "category": group.category,
                "mask_count": int(group.mask_count),
                "unique_point_count": int(group.unique_point_count),
            }
        )
    summary = {
        "source": "colmap_sparse_track_mask_association",
        "colmap_images_bin": str(colmap_images_bin),
        "grounded_masks_root": str(args.grounded_masks_root),
        "class_count": len(classes),
        "classes": classes,
        "selected_mask_count": int(selected_mask_count),
        "track_supported_mask_count": len(evidences),
        "assigned_mask_count": int(assigned_mask_count),
        "group_count": len(groups),
        "category_mask_counts": category_mask_counts,
        "class_mask_counts": class_mask_counts,
        "parameters": {
            "colmap_image_width": int(args.colmap_image_width),
            "colmap_image_height": int(args.colmap_image_height),
            "min_mask_score": float(args.min_mask_score),
            "min_area_ratio": float(args.min_area_ratio),
            "max_area_ratio": float(args.max_area_ratio),
            "min_track_points": int(args.min_track_points),
            "min_shared_points": int(args.min_shared_points),
            "min_overlap_ratio": float(args.min_overlap_ratio),
            "min_group_masks": int(args.min_group_masks),
        },
        "groups": [
            {
                "group_id": int(group.group_id),
                "class_name": class_names[group.group_id],
                "category": group.category,
                "mask_count": int(group.mask_count),
                "total_track_observations": int(group.total_track_observations),
                "unique_point_count": int(group.unique_point_count),
                "member_local_indices": list(group.member_local_indices),
            }
            for group in groups
        ],
        "frames": frame_summaries,
    }
    (args.output_root / "identity_mask_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(frame_summaries)} COLMAP-track instance frames with "
        f"{assigned_mask_count}/{selected_mask_count} assigned masks to {args.output_root}"
    )
    for class_entry in classes:
        print(f"  id={class_entry['global_id']} key={class_entry['raw_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
