"""Build 3D-first Gaussian cluster proposals from spatial connectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import (
    build_voxel_connected_components,
    filter_and_remap_components,
    load_gaussian_cloud,
    summarize_cluster_proposals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 3D Gaussian cluster proposals before semantic mask scoring."
    )
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="3D connectivity voxel edge length in scene units.",
    )
    parser.add_argument(
        "--min-voxel-count",
        type=int,
        default=1,
        help="Ignore occupied voxels with fewer Gaussians before connectivity.",
    )
    parser.add_argument(
        "--min-gaussians",
        type=int,
        default=100,
        help="Drop connected components smaller than this Gaussian count.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="Drop connected components larger than this Gaussian count.",
    )
    parser.add_argument(
        "--max-bbox-diag",
        type=float,
        default=None,
        help="Optional upper bound for cluster bbox diagonal in scene units.",
    )
    return parser.parse_args()


def _cluster_json_payload(
    cluster_ids: np.ndarray,
    xyz: np.ndarray,
    *,
    voxel_size: float,
    min_voxel_count: int,
    min_gaussians: int,
    max_gaussians: int | None,
    max_bbox_diag: float | None,
) -> dict:
    proposals = summarize_cluster_proposals(cluster_ids, xyz, voxel_size=voxel_size)
    if max_bbox_diag is not None:
        keep_ids = {
            proposal.global_object_id
            for proposal in proposals
            if proposal.bbox_diag <= max_bbox_diag
        }
        cluster_ids[~np.isin(cluster_ids, np.asarray(sorted(keep_ids), dtype=np.int32))] = -1
        cluster_ids[:] = filter_and_remap_components(
            cluster_ids,
            min_gaussians=1,
            max_gaussians=None,
        )
        proposals = summarize_cluster_proposals(cluster_ids, xyz, voxel_size=voxel_size)

    return {
        "group_count": len(proposals),
        "gaussian_count": int(cluster_ids.shape[0]),
        "assigned_gaussians": int(np.count_nonzero(cluster_ids >= 0)),
        "unknown_gaussians": int(np.count_nonzero(cluster_ids < 0)),
        "voxel_size": float(voxel_size),
        "min_voxel_count": int(min_voxel_count),
        "min_gaussians": int(min_gaussians),
        "max_gaussians": None if max_gaussians is None else int(max_gaussians),
        "max_bbox_diag": None if max_bbox_diag is None else float(max_bbox_diag),
        "groups": [
            {
                "global_object_id": proposal.global_object_id,
                "gaussian_count": proposal.gaussian_count,
                "voxel_count": proposal.voxel_count,
                "centroid_xyz": proposal.centroid_xyz.tolist(),
                "bbox_min_xyz": proposal.bbox_min_xyz.tolist(),
                "bbox_max_xyz": proposal.bbox_max_xyz.tolist(),
                "bbox_size_xyz": proposal.bbox_size_xyz.tolist(),
                "bbox_diag": proposal.bbox_diag,
                "support_frames": [],
                "total_vote_weight": 0.0,
            }
            for proposal in proposals
        ],
    }


def main() -> int:
    args = parse_args()
    cloud = load_gaussian_cloud(args.ply_path)
    xyz = cloud.xyz.astype(np.float32, copy=False)

    raw_component_ids = build_voxel_connected_components(
        xyz,
        voxel_size=args.voxel_size,
        min_voxel_count=args.min_voxel_count,
    )
    cluster_ids = filter_and_remap_components(
        raw_component_ids,
        min_gaussians=args.min_gaussians,
        max_gaussians=args.max_gaussians,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = _cluster_json_payload(
        cluster_ids,
        xyz,
        voxel_size=args.voxel_size,
        min_voxel_count=args.min_voxel_count,
        min_gaussians=args.min_gaussians,
        max_gaussians=args.max_gaussians,
        max_bbox_diag=args.max_bbox_diag,
    )
    np.save(args.output_root / "gaussian_cluster_ids.npy", cluster_ids.astype(np.int32))
    (args.output_root / "gaussian_clusters.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved {payload['group_count']} 3D clusters to {args.output_root} "
        f"({payload['assigned_gaussians']} / {payload['gaussian_count']} assigned)"
    )
    for group in payload["groups"][:20]:
        bbox = ", ".join(f"{value:.3f}" for value in group["bbox_size_xyz"])
        print(
            f"  cluster={group['global_object_id']} "
            f"gaussians={group['gaussian_count']} "
            f"bbox=({bbox}) diag={group['bbox_diag']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
