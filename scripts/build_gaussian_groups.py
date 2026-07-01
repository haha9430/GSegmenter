"""Build final Gaussian object assignments from multiview association results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.association import aggregate_local_instances, load_vote_evidence
from gsegmenter.mapping.gaussian_io import load_gaussian_cloud
from gsegmenter.mapping.grouping import (
    assign_gaussians_to_global_objects,
    save_gaussian_group_outputs,
    summarize_gaussian_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Gaussian groups.")
    parser.add_argument("--vote-evidence", type=Path, required=True)
    parser.add_argument("--association-manifest", type=Path, required=True)
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-vote-weight", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays = load_vote_evidence(args.vote_evidence)
    local_instances = aggregate_local_instances(*arrays)
    association_payload = json.loads(args.association_manifest.read_text(encoding="utf-8"))
    local_entries = association_payload["local_instances"]
    global_object_ids = [0] * len(local_entries)
    for entry in local_entries:
        global_object_ids[int(entry["local_index"])] = int(entry["global_object_id"])

    cloud = load_gaussian_cloud(args.ply_path)
    gaussian_object_ids = assign_gaussians_to_global_objects(
        local_instances,
        np.asarray(global_object_ids, dtype=np.int32),
        gaussian_count=cloud.vertex_count,
        min_vote_weight=args.min_vote_weight,
    )
    groups = summarize_gaussian_groups(
        gaussian_object_ids,
        cloud.xyz,
        local_instances,
        np.asarray(global_object_ids, dtype=np.int32),
    )
    save_gaussian_group_outputs(gaussian_object_ids, groups, args.output_root)
    print(f"Saved {len(groups)} Gaussian groups to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
