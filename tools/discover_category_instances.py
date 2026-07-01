"""Training-free first-pass category instance discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping.category_discovery import (
    accumulate_category_votes,
    build_category_instance_ids,
    default_category_specs,
    match_category,
    save_category_instance_summary,
    write_category_instance_highlight_ply,
)
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud, load_gaussian_table
from gsegmenter.mapping.lifting import (
    build_depth_consistency_mask,
    build_frame_vote_evidence,
    build_front_surface_mask,
    collect_mask_hits,
    load_frame_manifest_from_dir,
)
from gsegmenter.render.projection import project_world_points
from gsegmenter.segmentation.mask_io import FrameMasksManifest, MaskInstanceRecord, load_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift GroundingDINO+SAM2 masks onto an existing splat and cluster "
            "category votes into first-pass instance candidates without 3DGS training."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="colmap")
    parser.add_argument("--downscale-factor", type=int, default=2)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="ceil")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--include-categories", nargs="*", default=("tv", "chair", "table", "sofa", "storage"))
    parser.add_argument("--min-mask-score", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--use-opacity-weight", action="store_true")
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--front-surface-only", action="store_true")
    parser.add_argument("--front-surface-depth-margin", type=float, default=0.03)
    parser.add_argument("--min-vote-weight", type=float, default=0.5)
    parser.add_argument("--min-support-count", type=int, default=2)
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.0,
        help=(
            "Require object support / (object support + background support) to be at "
            "least this value. 0 disables explicit background filtering."
        ),
    )
    parser.add_argument("--voxel-size", type=float, default=0.035)
    parser.add_argument("--min-voxel-count", type=int, default=2)
    parser.add_argument("--min-gaussians", type=int, default=500)
    parser.add_argument("--max-instances-per-category", type=int, default=0)
    parser.add_argument("--dim-opacity-scale", type=float, default=0.25)
    parser.add_argument("--output-filename", type=str, default="category_instances.ply")
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


def _filter_manifest(
    manifest: FrameMasksManifest,
    payload: dict,
    *,
    category_by_instance: dict[int, int],
    min_score: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> FrameMasksManifest:
    width, height = manifest.image_size
    image_area = max(width * height, 1)
    kept: list[MaskInstanceRecord] = []
    raw_by_id = {int(instance["instance_id"]): instance for instance in payload.get("instances", [])}
    for instance in manifest.instances:
        if int(instance.instance_id) not in category_by_instance:
            continue
        raw_instance = raw_by_id.get(int(instance.instance_id), {})
        score = float(raw_instance.get("detection_score", instance.score))
        area_ratio = float(instance.area) / image_area
        if score < min_score:
            continue
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        kept.append(
            MaskInstanceRecord(
                instance_id=instance.instance_id,
                bbox_xyxy=instance.bbox_xyxy,
                score=score,
                area=instance.area,
                mask_path=instance.mask_path,
            )
        )
    return FrameMasksManifest(
        frame_index=manifest.frame_index,
        image_path=manifest.image_path,
        image_size=manifest.image_size,
        instances=tuple(kept),
    )


def _collect_background_indices(
    *,
    gaussian_xyz: np.ndarray,
    scene,
    frame,
    manifest: FrameMasksManifest,
    frame_dir: Path,
    depth_map: np.ndarray | None,
    front_surface_only: bool,
    front_surface_depth_margin: float,
) -> np.ndarray:
    """Collect visible Gaussians outside the selected object-mask union."""

    projection = project_world_points(gaussian_xyz, scene.intrinsics, frame)
    valid_mask = projection.valid_mask
    if front_surface_only:
        valid_mask = valid_mask & build_front_surface_mask(
            projection.image_points,
            projection.depths,
            projection.valid_mask,
            (scene.intrinsics.height, scene.intrinsics.width),
            depth_margin=front_surface_depth_margin,
        )
    if depth_map is not None:
        valid_mask = valid_mask & build_depth_consistency_mask(
            projection.image_points,
            projection.depths,
            projection.valid_mask,
            depth_map,
        )

    object_union = np.zeros((scene.intrinsics.height, scene.intrinsics.width), dtype=bool)
    for instance in manifest.instances:
        object_union |= load_binary_mask(frame_dir / instance.mask_path)

    visible_indices = np.flatnonzero(valid_mask).astype(np.int64)
    if visible_indices.size == 0:
        return visible_indices
    object_hits = collect_mask_hits(projection.image_points, valid_mask, object_union)
    if object_hits.size == 0:
        return visible_indices

    object_hit_mask = np.zeros((gaussian_xyz.shape[0],), dtype=bool)
    object_hit_mask[object_hits] = True
    return visible_indices[~object_hit_mask[visible_indices]]


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_specs = default_category_specs()
    requested = {str(name).casefold() for name in args.include_categories}
    specs = [spec for spec in all_specs if spec.name.casefold() in requested]
    if not specs:
        raise ValueError("No categories selected.")
    category_names = [spec.name for spec in specs]

    scene = _load_scene(args)
    cloud = load_gaussian_cloud(args.ply_path)
    gaussian_xyz = cloud.xyz
    table, header_properties = load_gaussian_table(args.ply_path)
    opacity_weights = None
    if args.use_opacity_weight and cloud.opacities is not None:
        opacity_weights = 1.0 / (1.0 + np.exp(-cloud.opacities[: len(gaussian_xyz)]))

    frames = scene.frames[: args.max_frames] if args.max_frames is not None else scene.frames
    all_evidences = []
    evidence_categories: list[int] = []
    background_support_counts = np.zeros((len(gaussian_xyz),), dtype=np.int32)
    selected_masks = 0
    vote_supporting_masks = 0
    category_mask_counts = {name: 0 for name in category_names}
    for frame in frames:
        frame_dir = args.masks_root / frame.file_path.stem
        manifest_path = frame_dir / "instances.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = load_frame_manifest_from_dir(frame_dir)
        raw_by_id = {int(instance["instance_id"]): instance for instance in payload.get("instances", [])}
        category_by_instance = {}
        for instance_id, raw_instance in raw_by_id.items():
            category_index = match_category(str(raw_instance.get("label", "")), specs)
            if category_index is not None:
                category_by_instance[int(instance_id)] = int(category_index)
        filtered_manifest = _filter_manifest(
            manifest,
            payload,
            category_by_instance=category_by_instance,
            min_score=args.min_mask_score,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )
        if not filtered_manifest.instances:
            continue

        depth_map = None
        if args.depth_root is not None:
            depth_path = args.depth_root / frame.file_path.stem / "depth.npy"
            if depth_path.exists():
                depth_map = np.load(depth_path).astype(np.float32)
        if args.min_foreground_ratio > 0.0:
            background_indices = _collect_background_indices(
                gaussian_xyz=gaussian_xyz,
                scene=scene,
                frame=frame,
                manifest=filtered_manifest,
                frame_dir=frame_dir,
                depth_map=depth_map,
                front_surface_only=args.front_surface_only,
                front_surface_depth_margin=args.front_surface_depth_margin,
            )
            if background_indices.size:
                np.add.at(
                    background_support_counts,
                    background_indices,
                    np.ones((background_indices.size,), dtype=np.int32),
                )
        evidences = build_frame_vote_evidence(
            gaussian_xyz=gaussian_xyz,
            intrinsics=scene.intrinsics,
            frame=frame,
            manifest=filtered_manifest,
            frame_dir=frame_dir,
            opacity_weights=opacity_weights,
            depth_map=depth_map,
            front_surface_only=args.front_surface_only,
            front_surface_depth_margin=args.front_surface_depth_margin,
        )
        selected_masks += len(filtered_manifest.instances)
        evidence_by_instance = {int(evidence.instance_id): evidence for evidence in evidences}
        for instance in filtered_manifest.instances:
            category_index = category_by_instance[int(instance.instance_id)]
            category_mask_counts[category_names[category_index]] += 1
            evidence = evidence_by_instance.get(int(instance.instance_id))
            if evidence is None:
                continue
            all_evidences.append(evidence)
            evidence_categories.append(category_index)
            vote_supporting_masks += 1

    category_votes, support_counts = accumulate_category_votes(
        all_evidences,
        evidence_categories,
        gaussian_count=len(gaussian_xyz),
        category_count=len(category_names),
    )
    instance_ids, proposals = build_category_instance_ids(
        xyz=gaussian_xyz,
        category_votes=category_votes,
        support_counts=support_counts,
        background_support_counts=background_support_counts if args.min_foreground_ratio > 0.0 else None,
        category_names=category_names,
        min_vote_weight=args.min_vote_weight,
        min_support_count=args.min_support_count,
        min_foreground_ratio=args.min_foreground_ratio,
        voxel_size=args.voxel_size,
        min_voxel_count=args.min_voxel_count,
        min_gaussians=args.min_gaussians,
        max_instances_per_category=args.max_instances_per_category,
    )

    output_path = args.output_root / args.output_filename
    write_category_instance_highlight_ply(
        output_path=output_path,
        table=table,
        header_properties=header_properties,
        instance_ids=instance_ids,
        dim_opacity_scale=args.dim_opacity_scale,
    )
    np.save(args.output_root / "gaussian_category_votes.npy", category_votes)
    np.save(args.output_root / "gaussian_category_support_counts.npy", support_counts)
    np.save(args.output_root / "gaussian_background_support_counts.npy", background_support_counts)
    np.save(args.output_root / "gaussian_instance_ids.npy", instance_ids)
    save_category_instance_summary(
        output_path=args.output_root / "category_instances.json",
        proposals=proposals,
        parameters={
            "dataset_root": str(args.dataset_root),
            "ply_path": str(args.ply_path),
            "masks_root": str(args.masks_root),
            "categories": category_names,
            "selected_masks": selected_masks,
            "vote_supporting_masks": vote_supporting_masks,
            "category_mask_counts": category_mask_counts,
            "min_mask_score": float(args.min_mask_score),
            "min_area_ratio": float(args.min_area_ratio),
            "max_area_ratio": float(args.max_area_ratio),
            "front_surface_only": bool(args.front_surface_only),
            "front_surface_depth_margin": float(args.front_surface_depth_margin),
            "min_vote_weight": float(args.min_vote_weight),
            "min_support_count": int(args.min_support_count),
            "min_foreground_ratio": float(args.min_foreground_ratio),
            "voxel_size": float(args.voxel_size),
            "min_voxel_count": int(args.min_voxel_count),
            "min_gaussians": int(args.min_gaussians),
        },
    )
    print(f"Wrote training-free category instance PLY to {output_path}")
    print(f"Selected {selected_masks} masks, {vote_supporting_masks} had Gaussian vote support")
    print(f"Found {len(proposals)} instance candidates")
    for proposal in proposals:
        print(
            f"  instance={proposal.instance_id} category={proposal.category} "
            f"rank={proposal.rank_in_category} gaussians={proposal.gaussian_count} "
            f"support_mean={proposal.support_count_mean:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
