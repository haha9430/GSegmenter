"""Print a compact summary of multiview association outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect association manifest outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    local_instances = payload["local_instances"]

    print(f"local_instance_count: {payload['local_instance_count']}")
    print(f"global_object_count: {payload['global_object_count']}")
    print(f"edge_count: {payload['edge_count']}")

    support_by_object: dict[int, int] = {}
    frames_by_object: dict[int, set[int]] = {}
    for entry in local_instances:
        object_id = int(entry["global_object_id"])
        support_by_object[object_id] = support_by_object.get(object_id, 0) + int(entry["support_size"])
        frames_by_object.setdefault(object_id, set()).add(int(entry["frame_index"]))

    top_objects = sorted(
        support_by_object.items(),
        key=lambda item: item[1],
        reverse=True,
    )[: args.top_k]

    print("")
    print("top_objects_by_support:")
    for object_id, support_size in top_objects:
        print(
            f"  object={object_id} "
            f"support={support_size} "
            f"frames={len(frames_by_object[object_id])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
