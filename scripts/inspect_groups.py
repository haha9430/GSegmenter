"""Print a compact summary of final Gaussian grouping outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Gaussian group outputs.")
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--object-ids", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.groups_json.read_text(encoding="utf-8"))
    object_ids = np.load(args.object_ids)

    assigned = int((object_ids >= 0).sum())
    unknown = int((object_ids < 0).sum())
    print(f"group_count: {payload['group_count']}")
    print(f"gaussian_count: {object_ids.shape[0]}")
    print(f"assigned_gaussians: {assigned}")
    print(f"unknown_gaussians: {unknown}")

    top_groups = sorted(
        payload["groups"],
        key=lambda group: int(group["gaussian_count"]),
        reverse=True,
    )[: args.top_k]

    print("")
    print("top_groups_by_gaussian_count:")
    for group in top_groups:
        print(
            f"  object={group['global_object_id']} "
            f"gaussians={group['gaussian_count']} "
            f"frames={len(group['support_frames'])} "
            f"vote_weight={group['total_vote_weight']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
