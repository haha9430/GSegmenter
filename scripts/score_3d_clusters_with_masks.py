"""Score 3D Gaussian clusters against multiview 2D semantic masks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.nerfstudio_scene import load_colmap_scene_with_nerfstudio_parser, load_nerfstudio_scene
from gsegmenter.mapping import (
    build_front_depth_buffer,
    filter_front_visible_points,
    infer_label_family,
    load_gaussian_cloud,
)
from gsegmenter.render.projection import project_world_points
from gsegmenter.segmentation.mask_io import load_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 3D cluster proposals into frames and score overlap with 2D masks."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--cluster-ids", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-format", choices=("nerfstudio", "colmap"), default="nerfstudio")
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--downscale-rounding-mode", choices=("floor", "round", "ceil"), default="floor")
    parser.add_argument("--images-path", type=Path, default=Path("images"))
    parser.add_argument("--colmap-path", type=Path, default=Path("colmap/sparse/0"))
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--include-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-points-per-cluster", type=int, default=5000)
    parser.add_argument("--min-visible-points", type=int, default=16)
    parser.add_argument("--min-hit-points", type=int, default=8)
    parser.add_argument("--min-hit-ratio", type=float, default=0.02)
    parser.add_argument(
        "--visibility-gate",
        action="store_true",
        help="Only score cluster points that are near the front-most Gaussian depth at each pixel.",
    )
    parser.add_argument(
        "--visibility-margin-ratio",
        type=float,
        default=0.05,
        help="Relative depth tolerance for visibility gating.",
    )
    parser.add_argument(
        "--visibility-min-margin",
        type=float,
        default=0.03,
        help="Minimum absolute depth tolerance for visibility gating in scene units.",
    )
    parser.add_argument(
        "--candidate-min-family-purity",
        type=float,
        default=0.60,
        help="Minimum best-family score ratio required for furniture_candidate=true.",
    )
    parser.add_argument(
        "--candidate-max-secondary-family-ratio",
        type=float,
        default=0.40,
        help="Maximum secondary/best family score ratio allowed for furniture_candidate=true.",
    )
    parser.add_argument(
        "--panel-min-diag",
        type=float,
        default=0.70,
        help="Minimum bbox diagonal for panel-like architectural rejection.",
    )
    parser.add_argument(
        "--panel-min-axis-ratio",
        type=float,
        default=0.45,
        help="Clusters with min_axis/max_axis below this ratio can be panel-like.",
    )
    parser.add_argument(
        "--panel-mid-axis-ratio",
        type=float,
        default=0.75,
        help="Clusters with mid_axis/max_axis below this ratio can be panel-like.",
    )
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


def _load_mask_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_ids_to_score(args: argparse.Namespace, cluster_ids: np.ndarray) -> list[int]:
    ids = {int(value) for value in np.unique(cluster_ids) if int(value) >= 0}
    if args.include_object_ids is not None:
        ids &= {int(value) for value in args.include_object_ids}
    if args.exclude_object_ids is not None:
        ids -= {int(value) for value in args.exclude_object_ids}
    return sorted(ids)


def _sample_cluster_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    if indices.size <= max_points:
        return indices
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def _geometry_tags(
    bbox_size_xyz: list[float] | None,
    bbox_diag: float | None,
    *,
    panel_min_diag: float,
    panel_min_axis_ratio: float,
    panel_mid_axis_ratio: float,
) -> list[str]:
    """Return simple shape tags for cluster-level proposal filtering."""

    if bbox_size_xyz is None or bbox_diag is None:
        return []
    dims = np.sort(np.asarray(bbox_size_xyz, dtype=np.float32))
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or dims[-1] <= 0.0:
        return []
    tags: list[str] = []
    min_ratio = float(dims[0] / dims[-1])
    mid_ratio = float(dims[1] / dims[-1])
    if (
        float(bbox_diag) >= panel_min_diag
        and min_ratio <= panel_min_axis_ratio
        and mid_ratio <= panel_mid_axis_ratio
    ):
        tags.append("panel_like")
    return tags


def _candidate_decision(
    family_scores: dict[str, float],
    geometry_tags: list[str],
    *,
    min_family_purity: float,
    max_secondary_family_ratio: float,
) -> tuple[bool, list[str], float, float]:
    """Decide whether a scored cluster is a plausible movable furniture object."""

    reasons: list[str] = []
    total_score = float(sum(max(0.0, value) for value in family_scores.values()))
    if total_score <= 0.0:
        return False, ["no_semantic_support"], 0.0, 0.0

    sorted_scores = sorted(family_scores.values(), reverse=True)
    best_score = float(sorted_scores[0])
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    purity = best_score / total_score
    secondary_ratio = second_score / best_score if best_score > 0.0 else 0.0
    if purity < min_family_purity:
        reasons.append("low_family_purity")
    if secondary_ratio > max_secondary_family_ratio:
        reasons.append("mixed_secondary_family")
    if "panel_like" in geometry_tags:
        reasons.append("panel_like_geometry")
    return len(reasons) == 0, reasons, purity, secondary_ratio


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")

    scene = _load_scene(args)
    cloud = load_gaussian_cloud(args.ply_path)
    cluster_ids = np.load(args.cluster_ids).astype(np.int32)
    if cluster_ids.shape[0] != cloud.vertex_count:
        raise ValueError(
            f"Cluster ID count {cluster_ids.shape[0]} does not match Gaussian count {cloud.vertex_count}"
        )
    groups_payload = json.loads(args.groups_json.read_text(encoding="utf-8"))
    groups_by_id = {
        int(group["global_object_id"]): group
        for group in groups_payload["groups"]
    }

    selected_cluster_ids = _cluster_ids_to_score(args, cluster_ids)
    selected_indices = {
        cluster_id: _sample_cluster_indices(
            np.flatnonzero(cluster_ids == cluster_id).astype(np.int64),
            args.max_points_per_cluster,
        )
        for cluster_id in selected_cluster_ids
    }

    scores: dict[int, dict[str, object]] = {}
    for cluster_id, indices in selected_indices.items():
        scores[cluster_id] = {
            "global_object_id": cluster_id,
            "sampled_gaussians": int(indices.size),
            "gaussian_count": int(groups_by_id.get(cluster_id, {}).get("gaussian_count", indices.size)),
            "bbox_size_xyz": groups_by_id.get(cluster_id, {}).get("bbox_size_xyz"),
            "bbox_diag": groups_by_id.get(cluster_id, {}).get("bbox_diag"),
            "visible_frames": 0,
            "label_scores": defaultdict(float),
            "label_families": defaultdict(float),
            "label_frame_support": defaultdict(int),
            "best_frames": [],
        }

    frames = scene.frames[:: args.frame_stride]
    xyz = cloud.xyz
    for frame in frames:
        frame_dir = args.masks_root / frame.file_path.stem
        manifest_path = frame_dir / "instances.json"
        if not manifest_path.exists():
            continue
        manifest = _load_mask_manifest(manifest_path)
        instances = manifest.get("instances", [])
        if not instances:
            continue
        masks = [
            (
                instance,
                load_binary_mask(frame_dir / str(instance["mask_path"])),
            )
            for instance in instances
        ]
        front_depth = None
        if args.visibility_gate:
            full_projection = project_world_points(xyz, scene.intrinsics, frame)
            front_depth = build_front_depth_buffer(
                full_projection.image_points,
                full_projection.depths,
                full_projection.valid_mask,
                height=scene.intrinsics.height,
                width=scene.intrinsics.width,
            )

        for cluster_id, indices in selected_indices.items():
            projection = project_world_points(xyz[indices], scene.intrinsics, frame)
            valid_local = np.flatnonzero(projection.valid_mask)
            if front_depth is not None:
                valid_local = filter_front_visible_points(
                    projection.image_points,
                    projection.depths,
                    valid_local,
                    front_depth,
                    margin_ratio=args.visibility_margin_ratio,
                    min_margin=args.visibility_min_margin,
                )
            visible_count = int(valid_local.size)
            if visible_count < args.min_visible_points:
                continue
            scores[cluster_id]["visible_frames"] = int(scores[cluster_id]["visible_frames"]) + 1

            pixel_x = np.floor(projection.image_points[valid_local, 0]).astype(np.int64)
            pixel_y = np.floor(projection.image_points[valid_local, 1]).astype(np.int64)
            best_frame_score = 0.0
            best_frame_label = None
            for instance, mask in masks:
                height, width = mask.shape
                in_bounds = (
                    (pixel_x >= 0)
                    & (pixel_x < width)
                    & (pixel_y >= 0)
                    & (pixel_y < height)
                )
                if not np.any(in_bounds):
                    continue
                hit_count = int(np.count_nonzero(mask[pixel_y[in_bounds], pixel_x[in_bounds]]))
                hit_ratio = hit_count / max(visible_count, 1)
                if hit_count < args.min_hit_points or hit_ratio < args.min_hit_ratio:
                    continue
                label = str(instance.get("label", "unknown"))
                family = infer_label_family(label)
                weight = hit_ratio * float(instance.get("score", 1.0))
                scores[cluster_id]["label_scores"][label] += weight
                scores[cluster_id]["label_families"][family] += weight
                scores[cluster_id]["label_frame_support"][label] += 1
                if weight > best_frame_score:
                    best_frame_score = float(weight)
                    best_frame_label = label
            if best_frame_label is not None:
                scores[cluster_id]["best_frames"].append(
                    {
                        "frame": frame.file_path.stem,
                        "label": best_frame_label,
                        "score": best_frame_score,
                        "visible_points": visible_count,
                    }
                )

    summary = []
    for cluster_id in selected_cluster_ids:
        record = scores[cluster_id]
        label_scores = dict(sorted(record["label_scores"].items(), key=lambda item: item[1], reverse=True))
        family_scores = dict(sorted(record["label_families"].items(), key=lambda item: item[1], reverse=True))
        best_label = next(iter(label_scores), None)
        best_family = next(iter(family_scores), None)
        geometry_tags = _geometry_tags(
            record["bbox_size_xyz"],
            record["bbox_diag"],
            panel_min_diag=args.panel_min_diag,
            panel_min_axis_ratio=args.panel_min_axis_ratio,
            panel_mid_axis_ratio=args.panel_mid_axis_ratio,
        )
        furniture_candidate, rejection_reasons, family_purity, secondary_family_ratio = _candidate_decision(
            family_scores,
            geometry_tags,
            min_family_purity=args.candidate_min_family_purity,
            max_secondary_family_ratio=args.candidate_max_secondary_family_ratio,
        )
        summary.append(
            {
                "global_object_id": cluster_id,
                "gaussian_count": record["gaussian_count"],
                "sampled_gaussians": record["sampled_gaussians"],
                "bbox_size_xyz": record["bbox_size_xyz"],
                "bbox_diag": record["bbox_diag"],
                "visible_frames": record["visible_frames"],
                "best_label": best_label,
                "best_label_family": best_family,
                "furniture_candidate": furniture_candidate,
                "rejection_reasons": rejection_reasons,
                "family_purity": family_purity,
                "secondary_family_ratio": secondary_family_ratio,
                "geometry_tags": geometry_tags,
                "label_scores": label_scores,
                "label_family_scores": family_scores,
                "label_frame_support": dict(record["label_frame_support"]),
                "best_frames": sorted(
                    record["best_frames"],
                    key=lambda item: float(item["score"]),
                    reverse=True,
                )[:8],
            }
        )
    summary.sort(
        key=lambda item: (
            max(item["label_family_scores"].values(), default=0.0),
            int(item["gaussian_count"]),
        ),
        reverse=True,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_count": len(summary),
        "frame_stride": int(args.frame_stride),
        "min_visible_points": int(args.min_visible_points),
        "min_hit_points": int(args.min_hit_points),
        "min_hit_ratio": float(args.min_hit_ratio),
        "visibility_gate": bool(args.visibility_gate),
        "visibility_margin_ratio": float(args.visibility_margin_ratio),
        "visibility_min_margin": float(args.visibility_min_margin),
        "candidate_min_family_purity": float(args.candidate_min_family_purity),
        "candidate_max_secondary_family_ratio": float(args.candidate_max_secondary_family_ratio),
        "panel_min_diag": float(args.panel_min_diag),
        "panel_min_axis_ratio": float(args.panel_min_axis_ratio),
        "panel_mid_axis_ratio": float(args.panel_mid_axis_ratio),
        "clusters": summary,
    }
    output_path = args.output_root / "semantic_cluster_scores.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved semantic cluster scores to {output_path}")
    for item in summary[:20]:
        family_score = max(item["label_family_scores"].values(), default=0.0)
        print(
            f"  cluster={item['global_object_id']} gaussians={item['gaussian_count']} "
            f"best={item['best_label']} family={item['best_label_family']} score={family_score:.3f} "
            f"candidate={item['furniture_candidate']} reasons={','.join(item['rejection_reasons']) or '-'} "
            f"visible_frames={item['visible_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
