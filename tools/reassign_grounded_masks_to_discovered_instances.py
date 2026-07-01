"""Re-label dense Grounded-SAM masks with discovered 3D instance ids."""

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

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping.category_discovery import default_category_specs, match_category
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud
from gsegmenter.mapping.lifting import build_frame_vote_evidence, load_frame_manifest_from_dir
from gsegmenter.segmentation.mask_io import FrameMasksManifest, MaskInstanceRecord, save_binary_mask, save_frame_masks_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign original GroundingDINO+SAM2 dense masks to first-pass 3D "
            "instance candidates, then write second-pass identity training masks."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--grounded-masks-root", type=Path, required=True)
    parser.add_argument("--instance-ids", type=Path, required=True)
    parser.add_argument("--instances-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="colmap")
    parser.add_argument("--downscale-factor", type=int, default=2)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="ceil")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--include-categories", nargs="*", default=("tv", "chair", "table", "sofa", "storage"))
    parser.add_argument("--include-instance-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-instance-ids", type=int, nargs="*", default=None)
    parser.add_argument("--min-mask-score", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--min-instance-hit-ratio", type=float, default=0.20)
    parser.add_argument("--min-instance-hit-count", type=int, default=32)
    parser.add_argument("--front-surface-only", action="store_true")
    parser.add_argument("--front-surface-depth-margin", type=float, default=0.03)
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--background-score", type=float, default=0.0)
    parser.add_argument("--include-empty-background-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_scene(args: argparse.Namespace):
    if args.scene_format == "colmap":
        return load_colmap_scene_with_nerfstudio_parser(
            args.dataset_root,
            downscale_factor=args.downscale_factor,
            downscale_rounding_mode=args.downscale_rounding_mode,
            images_path=args.images_path,
            colmap_path=args.colmap_path,
        )
    return load_nerfstudio_scene(args.dataset_root)


def _load_discovered_instances(instances_json: Path) -> list[dict]:
    payload = json.loads(Path(instances_json).read_text(encoding="utf-8"))
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        raise ValueError(f"{instances_json} does not contain an instances list.")
    return instances


def _instance_class_name(instance: dict) -> str:
    category = str(instance["category"])
    rank = int(instance.get("rank_in_category", instance["instance_id"]))
    return f"{category}_{rank + 1:02d}"


def _select_discovered_instances(
    instances: list[dict],
    *,
    include_ids: set[int] | None,
    exclude_ids: set[int],
) -> list[dict]:
    selected = []
    for instance in instances:
        instance_id = int(instance["instance_id"])
        if include_ids is not None and instance_id not in include_ids:
            continue
        if instance_id in exclude_ids:
            continue
        selected.append(instance)
    selected.sort(key=lambda item: int(item["instance_id"]))
    if not selected:
        raise ValueError("No discovered instances were selected.")
    return selected


def _candidate_score(instance: dict) -> float:
    if instance.get("detection_score") is not None:
        return float(instance["detection_score"])
    return float(instance.get("score", 0.0))


def choose_instance_for_mask(
    gaussian_indices: np.ndarray,
    instance_ids: np.ndarray,
    *,
    allowed_instance_ids: set[int],
    min_hit_ratio: float,
    min_hit_count: int,
) -> tuple[int | None, int, float]:
    """Choose the discovered instance with strongest support inside one 2D mask."""

    if gaussian_indices.size == 0:
        return None, 0, 0.0
    lifted_ids = instance_ids[gaussian_indices]
    keep = np.isin(lifted_ids, np.asarray(sorted(allowed_instance_ids), dtype=np.int32))
    lifted_ids = lifted_ids[keep]
    if lifted_ids.size == 0:
        return None, 0, 0.0
    values, counts = np.unique(lifted_ids, return_counts=True)
    best_index = int(np.argmax(counts))
    best_instance_id = int(values[best_index])
    best_count = int(counts[best_index])
    hit_ratio = best_count / float(max(int(gaussian_indices.size), 1))
    if best_count < int(min_hit_count) or hit_ratio < float(min_hit_ratio):
        return None, best_count, hit_ratio
    return best_instance_id, best_count, hit_ratio


def _filtered_manifest_and_metadata(
    *,
    manifest: FrameMasksManifest,
    payload: dict,
    specs,
    discovered_by_category: dict[str, set[int]],
    min_score: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[FrameMasksManifest, dict[int, dict]]:
    width, height = manifest.image_size
    image_area = max(width * height, 1)
    raw_by_id = {int(instance["instance_id"]): instance for instance in payload.get("instances", [])}
    records = []
    metadata: dict[int, dict] = {}
    for instance in manifest.instances:
        raw = raw_by_id.get(int(instance.instance_id), {})
        category_index = match_category(str(raw.get("label", "")), specs)
        if category_index is None:
            continue
        category_name = specs[category_index].name
        if category_name not in discovered_by_category:
            continue
        score = _candidate_score(raw)
        area_ratio = float(instance.area) / image_area
        if score < min_score or area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        records.append(
            MaskInstanceRecord(
                instance_id=instance.instance_id,
                bbox_xyxy=instance.bbox_xyxy,
                score=score,
                area=instance.area,
                mask_path=instance.mask_path,
            )
        )
        metadata[int(instance.instance_id)] = {
            "category": category_name,
            "label": str(raw.get("label", "")),
            "candidate_score": score,
            "area_ratio": area_ratio,
        }
    return (
        FrameMasksManifest(
            frame_index=manifest.frame_index,
            image_path=manifest.image_path,
            image_size=manifest.image_size,
            instances=tuple(records),
        ),
        metadata,
    )


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    scene = _load_scene(args)
    cloud = load_gaussian_cloud(args.ply_path)
    gaussian_instance_ids = np.load(args.instance_ids).astype(np.int32)
    if gaussian_instance_ids.shape != (cloud.vertex_count,):
        raise ValueError(
            f"instance id count {gaussian_instance_ids.shape[0]} does not match Gaussian count {cloud.vertex_count}"
        )

    discovered = _select_discovered_instances(
        _load_discovered_instances(args.instances_json),
        include_ids=None if args.include_instance_ids is None else {int(value) for value in args.include_instance_ids},
        exclude_ids={int(value) for value in (args.exclude_instance_ids or [])},
    )
    class_by_instance_id = {
        int(instance["instance_id"]): class_index
        for class_index, instance in enumerate(discovered, start=1)
    }
    name_by_instance_id = {int(instance["instance_id"]): _instance_class_name(instance) for instance in discovered}
    category_by_instance_id = {int(instance["instance_id"]): str(instance["category"]) for instance in discovered}
    discovered_by_category: dict[str, set[int]] = {}
    for instance_id, category in category_by_instance_id.items():
        discovered_by_category.setdefault(category, set()).add(instance_id)

    all_specs = default_category_specs()
    requested_categories = {str(value).casefold() for value in args.include_categories}
    specs = [spec for spec in all_specs if spec.name.casefold() in requested_categories]
    if not specs:
        raise ValueError("No categories selected.")

    frames = scene.frames[: args.max_frames] if args.max_frames is not None else scene.frames
    frame_summaries = []
    class_mask_counts = {name_by_instance_id[int(instance["instance_id"])]: 0 for instance in discovered}
    selected_dense_masks = 0
    assigned_dense_masks = 0
    for output_frame_index, frame in enumerate(frames):
        frame_dir = args.grounded_masks_root / frame.file_path.stem
        manifest_path = frame_dir / "instances.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = load_frame_manifest_from_dir(frame_dir)
        filtered_manifest, metadata = _filtered_manifest_and_metadata(
            manifest=manifest,
            payload=payload,
            specs=specs,
            discovered_by_category=discovered_by_category,
            min_score=args.min_mask_score,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )
        if not filtered_manifest.instances:
            continue
        selected_dense_masks += len(filtered_manifest.instances)

        depth_map = None
        if args.depth_root is not None:
            depth_path = args.depth_root / frame.file_path.stem / "depth.npy"
            if depth_path.exists():
                depth_map = np.load(depth_path).astype(np.float32)
        evidences = build_frame_vote_evidence(
            gaussian_xyz=cloud.xyz,
            intrinsics=scene.intrinsics,
            frame=frame,
            manifest=filtered_manifest,
            frame_dir=frame_dir,
            depth_map=depth_map,
            front_surface_only=args.front_surface_only,
            front_surface_depth_margin=args.front_surface_depth_margin,
        )
        evidence_by_mask_id = {int(evidence.instance_id): evidence for evidence in evidences}

        output_frame_dir = args.output_root / f"frame_{output_frame_index:05d}"
        output_records = [
            MaskInstanceRecord(
                instance_id=int(args.background_id),
                bbox_xyxy=(0, 0, scene.intrinsics.width - 1, scene.intrinsics.height - 1),
                score=float(args.background_score),
                area=int(scene.intrinsics.width * scene.intrinsics.height),
                mask_path="mask_background.png",
            )
        ]
        save_binary_mask(
            np.ones((scene.intrinsics.height, scene.intrinsics.width), dtype=bool),
            output_frame_dir / "mask_background.png",
        )

        assigned_summary = []
        output_mask_index = 0
        for record in filtered_manifest.instances:
            info = metadata[int(record.instance_id)]
            allowed = discovered_by_category[str(info["category"])]
            evidence = evidence_by_mask_id.get(int(record.instance_id))
            if evidence is None:
                continue
            chosen_id, hit_count, hit_ratio = choose_instance_for_mask(
                evidence.gaussian_indices,
                gaussian_instance_ids,
                allowed_instance_ids=allowed,
                min_hit_ratio=args.min_instance_hit_ratio,
                min_hit_count=args.min_instance_hit_count,
            )
            if chosen_id is None:
                continue
            mask_name = f"mask_{output_mask_index:04d}.png"
            shutil.copy2(frame_dir / record.mask_path, output_frame_dir / mask_name)
            class_id = int(class_by_instance_id[chosen_id])
            output_records.append(
                MaskInstanceRecord(
                    instance_id=class_id,
                    bbox_xyxy=record.bbox_xyxy,
                    score=float(record.score),
                    area=int(record.area),
                    mask_path=mask_name,
                )
            )
            output_mask_index += 1
            assigned_dense_masks += 1
            class_mask_counts[name_by_instance_id[chosen_id]] += 1
            assigned_summary.append(
                {
                    "source_instance_id": int(record.instance_id),
                    "source_mask_path": record.mask_path,
                    "source_label": info["label"],
                    "assigned_instance_id": int(chosen_id),
                    "class_id": class_id,
                    "class_name": name_by_instance_id[chosen_id],
                    "hit_count": int(hit_count),
                    "hit_ratio": float(hit_ratio),
                    "area": int(record.area),
                    "bbox_xyxy": list(record.bbox_xyxy),
                }
            )

        if len(output_records) == 1 and not args.include_empty_background_frames:
            for path in output_frame_dir.glob("*"):
                path.unlink()
            output_frame_dir.rmdir()
            continue

        save_frame_masks_manifest(
            FrameMasksManifest(
                frame_index=output_frame_index,
                image_path=str(frame.file_path),
                image_size=(scene.intrinsics.width, scene.intrinsics.height),
                instances=tuple(output_records),
            ),
            output_frame_dir / "instances.json",
        )
        frame_summaries.append(
            {
                "output_frame": output_frame_dir.name,
                "image_path": str(frame.file_path),
                "assigned_mask_count": len(output_records) - 1,
                "assigned": assigned_summary,
            }
        )

    classes = [{"raw_key": "background", "global_id": int(args.background_id)}]
    for instance in discovered:
        source_id = int(instance["instance_id"])
        classes.append(
            {
                "raw_key": name_by_instance_id[source_id],
                "global_id": int(class_by_instance_id[source_id]),
                "source_instance_id": source_id,
                "category": category_by_instance_id[source_id],
            }
        )
    summary = {
        "source": "grounded_sam_dense_masks_reassigned_to_discovered_instances",
        "grounded_masks_root": str(args.grounded_masks_root),
        "ply_path": str(args.ply_path),
        "instance_ids": str(args.instance_ids),
        "instances_json": str(args.instances_json),
        "class_count": len(classes),
        "classes": classes,
        "frame_count": len(frame_summaries),
        "selected_dense_masks": int(selected_dense_masks),
        "assigned_dense_masks": int(assigned_dense_masks),
        "class_mask_counts": class_mask_counts,
        "min_instance_hit_ratio": float(args.min_instance_hit_ratio),
        "min_instance_hit_count": int(args.min_instance_hit_count),
        "front_surface_only": bool(args.front_surface_only),
        "front_surface_depth_margin": float(args.front_surface_depth_margin),
        "frames": frame_summaries,
    }
    (args.output_root / "identity_mask_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(frame_summaries)} dense reassigned-mask frames with "
        f"{assigned_dense_masks}/{selected_dense_masks} assigned masks to {args.output_root}"
    )
    for class_entry in classes:
        print(f"  id={class_entry['global_id']} key={class_entry['raw_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
