"""Project discovered Gaussian instances into 2D masks for identity training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud
from gsegmenter.mapping.lifting import build_front_surface_mask
from gsegmenter.render.projection import project_world_points
from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    save_binary_mask,
    save_frame_masks_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert first-pass Gaussian instance ids into frame-wise instance masks "
            "that can be used as second-pass identity-splatfacto supervision."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--instance-ids", type=Path, required=True)
    parser.add_argument("--instances-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="colmap")
    parser.add_argument("--downscale-factor", type=int, default=2)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="ceil")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--include-instance-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-instance-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--front-surface-only", action="store_true")
    parser.add_argument("--front-surface-depth-margin", type=float, default=0.03)
    parser.add_argument("--dilation-radius", type=int, default=4)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--background-score", type=float, default=0.0)
    parser.add_argument("--instance-score", type=float, default=1.0)
    parser.add_argument("--include-empty-background-frames", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
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


def _load_instance_records(instances_json: Path) -> list[dict]:
    payload = json.loads(Path(instances_json).read_text(encoding="utf-8"))
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        raise ValueError(f"{instances_json} does not contain an instances list.")
    return instances


def _instance_class_name(instance: dict) -> str:
    category = str(instance["category"])
    rank = int(instance.get("rank_in_category", instance["instance_id"]))
    return f"{category}_{rank + 1:02d}"


def _select_instances(
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
        raise ValueError("No discovered instances were selected for projection.")
    return selected


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return np.asarray(dilated, dtype=np.uint8) > 0


def rasterize_projected_points(
    image_points: np.ndarray,
    gaussian_mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    dilation_radius: int,
) -> np.ndarray:
    """Rasterize projected Gaussian centers into a dense-ish binary mask."""

    height, width = (int(value) for value in image_shape)
    mask = np.zeros((height, width), dtype=bool)
    indices = np.flatnonzero(gaussian_mask)
    if indices.size == 0:
        return mask
    pixel_x = np.floor(image_points[indices, 0]).astype(np.int64)
    pixel_y = np.floor(image_points[indices, 1]).astype(np.int64)
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    if not np.any(in_bounds):
        return mask
    mask[pixel_y[in_bounds], pixel_x[in_bounds]] = True
    return _dilate_mask(mask, int(dilation_radius))


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y_indices, x_indices = np.nonzero(mask)
    if x_indices.size == 0:
        return (0, 0, 0, 0)
    return (
        int(x_indices.min()),
        int(y_indices.min()),
        int(x_indices.max()),
        int(y_indices.max()),
    )


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    scene = _load_scene(args)
    cloud = load_gaussian_cloud(args.ply_path)
    instance_ids = np.load(args.instance_ids).astype(np.int32)
    if instance_ids.shape != (cloud.vertex_count,):
        raise ValueError(
            f"instance id count {instance_ids.shape[0]} does not match Gaussian count {cloud.vertex_count}"
        )

    discovered_instances = _select_instances(
        _load_instance_records(args.instances_json),
        include_ids=None if args.include_instance_ids is None else {int(value) for value in args.include_instance_ids},
        exclude_ids={int(value) for value in (args.exclude_instance_ids or [])},
    )
    class_by_instance_id = {
        int(instance["instance_id"]): class_index
        for class_index, instance in enumerate(discovered_instances, start=1)
    }
    class_names = {
        int(instance["instance_id"]): _instance_class_name(instance)
        for instance in discovered_instances
    }

    frames = scene.frames[: args.max_frames] if args.max_frames is not None else scene.frames
    frame_summaries = []
    total_instance_masks = 0
    class_mask_counts = {class_names[int(instance["instance_id"])]: 0 for instance in discovered_instances}
    class_name_by_class_id = {
        int(class_by_instance_id[int(instance["instance_id"])]): class_names[int(instance["instance_id"])]
        for instance in discovered_instances
    }
    for output_frame_index, frame in enumerate(frames):
        output_frame_dir = args.output_root / f"frame_{output_frame_index:05d}"
        manifest_path = output_frame_dir / "instances.json"
        if args.skip_existing and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            instance_count = max(0, len(manifest.get("instances", [])) - 1)
            total_instance_masks += instance_count
            for instance in manifest.get("instances", []):
                class_id = int(instance.get("instance_id", -1))
                class_name = class_name_by_class_id.get(class_id)
                if class_name is not None:
                    class_mask_counts[class_name] += 1
            frame_summaries.append(
                {
                    "output_frame": output_frame_dir.name,
                    "image_path": str(frame.file_path),
                    "instance_mask_count": instance_count,
                    "selected": [],
                    "skipped_existing": True,
                }
            )
            continue
        projection = project_world_points(cloud.xyz, scene.intrinsics, frame)
        valid_mask = projection.valid_mask
        if args.front_surface_only:
            valid_mask = valid_mask & build_front_surface_mask(
                projection.image_points,
                projection.depths,
                projection.valid_mask,
                (scene.intrinsics.height, scene.intrinsics.width),
                depth_margin=args.front_surface_depth_margin,
            )

        records = [
            MaskInstanceRecord(
                instance_id=int(args.background_id),
                bbox_xyxy=(0, 0, scene.intrinsics.width - 1, scene.intrinsics.height - 1),
                score=float(args.background_score),
                area=int(scene.intrinsics.width * scene.intrinsics.height),
                mask_path="mask_background.png",
            )
        ]
        selected_summary = []
        save_binary_mask(
            np.ones((scene.intrinsics.height, scene.intrinsics.width), dtype=bool),
            output_frame_dir / "mask_background.png",
        )

        mask_index = 0
        for instance in discovered_instances:
            source_instance_id = int(instance["instance_id"])
            gaussian_mask = valid_mask & (instance_ids == source_instance_id)
            mask = rasterize_projected_points(
                projection.image_points,
                gaussian_mask,
                image_shape=(scene.intrinsics.height, scene.intrinsics.width),
                dilation_radius=args.dilation_radius,
            )
            area = int(mask.sum())
            if area < int(args.min_mask_pixels):
                continue
            mask_name = f"mask_{mask_index:04d}.png"
            save_binary_mask(mask, output_frame_dir / mask_name)
            class_id = class_by_instance_id[source_instance_id]
            records.append(
                MaskInstanceRecord(
                    instance_id=class_id,
                    bbox_xyxy=_mask_bbox(mask),
                    score=float(args.instance_score),
                    area=area,
                    mask_path=mask_name,
                )
            )
            mask_index += 1
            total_instance_masks += 1
            class_mask_counts[class_names[source_instance_id]] += 1
            selected_summary.append(
                {
                    "source_instance_id": source_instance_id,
                    "class_id": class_id,
                    "class_name": class_names[source_instance_id],
                    "area": area,
                    "bbox_xyxy": list(_mask_bbox(mask)),
                }
            )

        if len(records) == 1 and not args.include_empty_background_frames:
            for path in output_frame_dir.glob("*"):
                path.unlink()
            output_frame_dir.rmdir()
            continue

        manifest = FrameMasksManifest(
            frame_index=output_frame_index,
            image_path=str(frame.file_path),
            image_size=(scene.intrinsics.width, scene.intrinsics.height),
            instances=tuple(records),
        )
        save_frame_masks_manifest(manifest, output_frame_dir / "instances.json")
        frame_summaries.append(
            {
                "output_frame": output_frame_dir.name,
                "image_path": str(frame.file_path),
                "instance_mask_count": len(records) - 1,
                "selected": selected_summary,
            }
        )

    classes = [{"raw_key": "background", "global_id": int(args.background_id)}]
    for instance in discovered_instances:
        source_instance_id = int(instance["instance_id"])
        classes.append(
            {
                "raw_key": class_names[source_instance_id],
                "global_id": int(class_by_instance_id[source_instance_id]),
                "source_instance_id": source_instance_id,
                "category": str(instance["category"]),
            }
        )
    summary = {
        "source": "projected_discovered_gaussian_instances",
        "ply_path": str(args.ply_path),
        "instance_ids": str(args.instance_ids),
        "instances_json": str(args.instances_json),
        "class_count": len(classes),
        "classes": classes,
        "frame_count": len(frame_summaries),
        "total_instance_masks": total_instance_masks,
        "class_mask_counts": class_mask_counts,
        "dilation_radius": int(args.dilation_radius),
        "min_mask_pixels": int(args.min_mask_pixels),
        "front_surface_only": bool(args.front_surface_only),
        "front_surface_depth_margin": float(args.front_surface_depth_margin),
        "frames": frame_summaries,
    }
    (args.output_root / "identity_mask_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(frame_summaries)} projected-mask frames with "
        f"{total_instance_masks} instance masks to {args.output_root}"
    )
    for class_entry in classes:
        print(f"  id={class_entry['global_id']} key={class_entry['raw_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
