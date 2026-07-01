"""Discover 3D object proposals from grounded 2D masks and Gaussian votes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping.association import (
    aggregate_local_instances,
    assign_global_objects,
    build_association_pairs,
    infer_label_family,
)
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud, load_gaussian_table
from gsegmenter.mapping.grouping import assign_gaussians_to_global_objects
from gsegmenter.mapping.lifting import (
    build_frame_vote_evidence,
    load_frame_manifest_from_dir,
    save_vote_evidence,
    save_vote_summary,
)
from gsegmenter.mapping.object_proposals import (
    save_object_proposals,
    select_top_proposals,
    summarize_object_proposals,
    write_proposal_highlight_ply,
)
from gsegmenter.segmentation.mask_io import FrameMasksManifest, MaskInstanceRecord


DEFAULT_FURNITURE_REGEX = (
    r"sofa|couch|chair|armchair|stool|ottoman|table|desk|nightstand|"
    r"television|tv|speaker|cabinet|shelf|bookshelf|wardrobe|drawer|dresser|"
    r"bed|lamp|plant"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build first-pass object proposals by lifting grounded 2D furniture masks "
            "into Gaussian space and associating overlapping frame-local instances."
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
    parser.add_argument("--label-regex", type=str, default=DEFAULT_FURNITURE_REGEX)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--use-opacity-weight", action="store_true")
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument(
        "--front-surface-only",
        action="store_true",
        help="Use z-buffer style per-pixel front-surface filtering before collecting mask votes.",
    )
    parser.add_argument("--front-surface-depth-margin", type=float, default=0.03)
    parser.add_argument("--max-frame-gap", type=int, default=3)
    parser.add_argument("--min-shared-gaussians", type=int, default=96)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.08)
    parser.add_argument("--min-support-size", type=int, default=64)
    parser.add_argument("--max-support-size", type=int, default=None)
    parser.add_argument("--min-proposal-gaussians", type=int, default=512)
    parser.add_argument("--min-proposal-frames", type=int, default=2)
    parser.add_argument("--highlight-top-k", type=int, default=24)
    parser.add_argument("--dim-opacity-scale", type=float, default=0.35)
    parser.add_argument("--allow-cross-family", action="store_true")
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


def _filter_manifest_from_payload(
    manifest_path: Path,
    *,
    label_pattern: re.Pattern[str],
    min_score: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[FrameMasksManifest, dict[tuple[int, int], tuple[str | None, str | None]]]:
    manifest = load_frame_manifest_from_dir(manifest_path.parent)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_instance_id = {int(instance["instance_id"]): instance for instance in payload.get("instances", [])}
    width, height = manifest.image_size
    image_area = max(width * height, 1)
    kept = []
    metadata: dict[tuple[int, int], tuple[str | None, str | None]] = {}
    for instance in manifest.instances:
        raw_instance = by_instance_id.get(int(instance.instance_id), {})
        label = raw_instance.get("label")
        label_text = str(label) if label is not None else None
        if label_text is None or label_pattern.search(label_text) is None:
            continue
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
        metadata[(manifest.frame_index, int(instance.instance_id))] = (
            label_text,
            infer_label_family(label_text),
        )
    return (
        FrameMasksManifest(
            frame_index=manifest.frame_index,
            image_path=manifest.image_path,
            image_size=manifest.image_size,
            instances=tuple(kept),
        ),
        metadata,
    )


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    label_pattern = re.compile(args.label_regex, flags=re.IGNORECASE)

    scene = _load_scene(args)
    cloud = load_gaussian_cloud(args.ply_path)
    gaussian_xyz = cloud.xyz
    if args.max_gaussians is not None:
        gaussian_xyz = gaussian_xyz[: args.max_gaussians]
    opacity_weights = None
    if args.use_opacity_weight and cloud.opacities is not None:
        opacity_weights = 1.0 / (1.0 + np.exp(-cloud.opacities[: len(gaussian_xyz)]))

    frames = scene.frames[: args.max_frames] if args.max_frames is not None else scene.frames
    all_evidences = []
    label_metadata: dict[tuple[int, int], tuple[str | None, str | None]] = {}
    selected_frame_instances = 0
    for frame in frames:
        frame_dir = args.masks_root / frame.file_path.stem
        manifest_path = frame_dir / "instances.json"
        if not manifest_path.exists():
            continue
        manifest, frame_metadata = _filter_manifest_from_payload(
            manifest_path,
            label_pattern=label_pattern,
            min_score=args.min_score,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )
        if not manifest.instances:
            continue
        depth_map = None
        if args.depth_root is not None:
            depth_path = args.depth_root / frame.file_path.stem / "depth.npy"
            if depth_path.exists():
                depth_map = np.load(depth_path).astype(np.float32)
        evidences = build_frame_vote_evidence(
            gaussian_xyz=gaussian_xyz,
            intrinsics=scene.intrinsics,
            frame=frame,
            manifest=manifest,
            frame_dir=frame_dir,
            opacity_weights=opacity_weights,
            depth_map=depth_map,
            front_surface_only=args.front_surface_only,
            front_surface_depth_margin=args.front_surface_depth_margin,
        )
        all_evidences.extend(evidences)
        label_metadata.update(frame_metadata)
        selected_frame_instances += len(manifest.instances)
        print(
            f"Processed {frame.file_path.name}: "
            f"{len(manifest.instances)} furniture masks, {len(evidences)} vote-supporting masks"
        )

    vote_evidence_path = args.output_root / "vote_evidence.npz"
    save_vote_evidence(all_evidences, vote_evidence_path)
    save_vote_summary(all_evidences, args.output_root / "vote_summary.json", total_gaussians=len(gaussian_xyz))

    arrays = (
        np.concatenate(
            [np.full((len(e.gaussian_indices),), e.frame_index, dtype=np.int32) for e in all_evidences]
        )
        if all_evidences
        else np.zeros((0,), dtype=np.int32),
        np.concatenate(
            [np.full((len(e.gaussian_indices),), e.instance_id, dtype=np.int32) for e in all_evidences]
        )
        if all_evidences
        else np.zeros((0,), dtype=np.int32),
        np.concatenate([e.gaussian_indices for e in all_evidences]) if all_evidences else np.zeros((0,), dtype=np.int64),
        np.concatenate([e.weights for e in all_evidences]) if all_evidences else np.zeros((0,), dtype=np.float32),
    )
    local_instances = aggregate_local_instances(*arrays)
    for local_instance in local_instances:
        label, label_family = label_metadata.get((local_instance.frame_index, local_instance.instance_id), (None, None))
        local_instance.label = label
        local_instance.label_family = label_family

    candidate_instances = [
        instance
        for instance in local_instances
        if instance.support_size >= args.min_support_size
        and (args.max_support_size is None or instance.support_size <= args.max_support_size)
    ]
    pairs = build_association_pairs(
        candidate_instances,
        max_frame_gap=args.max_frame_gap,
        min_shared_gaussians=args.min_shared_gaussians,
        min_overlap_ratio=args.min_overlap_ratio,
        require_same_label_family=not args.allow_cross_family,
    )
    global_object_ids = assign_global_objects(candidate_instances, pairs, total_local_count=len(local_instances))
    gaussian_object_ids = assign_gaussians_to_global_objects(
        local_instances,
        global_object_ids,
        gaussian_count=len(gaussian_xyz),
    )
    proposals = summarize_object_proposals(
        gaussian_object_ids=gaussian_object_ids,
        gaussian_xyz=gaussian_xyz,
        local_instances=local_instances,
        global_object_ids=global_object_ids,
    )
    proposals.sort(key=lambda proposal: proposal.gaussian_count, reverse=True)

    np.save(args.output_root / "proposal_object_ids.npy", gaussian_object_ids.astype(np.int32))
    save_object_proposals(
        proposals=proposals,
        output_path=args.output_root / "object_proposals.json",
        parameters={
            "selected_frame_instances": selected_frame_instances,
            "vote_supporting_instances": len(local_instances),
            "candidate_instances": len(candidate_instances),
            "edge_count": len(pairs),
            "label_regex": args.label_regex,
            "min_score": args.min_score,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "max_frame_gap": args.max_frame_gap,
            "min_shared_gaussians": args.min_shared_gaussians,
            "min_overlap_ratio": args.min_overlap_ratio,
            "require_same_label_family": not args.allow_cross_family,
            "front_surface_only": args.front_surface_only,
            "front_surface_depth_margin": args.front_surface_depth_margin,
        },
    )

    table, header_properties = load_gaussian_table(args.ply_path)
    keep_ids = select_top_proposals(
        proposals,
        limit=args.highlight_top_k,
        min_gaussians=args.min_proposal_gaussians,
        min_support_frames=args.min_proposal_frames,
    )
    write_proposal_highlight_ply(
        output_path=args.output_root / "object_proposals_highlighted.ply",
        table=table[: len(gaussian_object_ids)],
        header_properties=header_properties,
        proposal_ids=gaussian_object_ids,
        keep_proposal_ids=keep_ids,
        dim_opacity_scale=args.dim_opacity_scale,
    )
    print(
        f"Discovered {len(proposals)} proposals from {len(local_instances)} vote-supporting local instances. "
        f"Highlighted {len(keep_ids)} proposals in {args.output_root / 'object_proposals_highlighted.ply'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
