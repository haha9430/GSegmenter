"""Associate frame-local mask instances into global object hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.association import (
    aggregate_local_instances,
    assign_global_objects,
    build_association_pairs,
    infer_label_family,
    load_vote_evidence,
    save_association_edges,
    save_association_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Associate local instances across views.")
    parser.add_argument("--vote-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frame-gap", type=int, default=1)
    parser.add_argument("--min-shared-gaussians", type=int, default=32)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--min-support-size", type=int, default=1)
    parser.add_argument("--max-support-size", type=int, default=None)
    parser.add_argument("--masks-root", type=Path, default=None)
    parser.add_argument("--require-same-label-family", action="store_true")
    return parser.parse_args()


def _attach_label_metadata(local_instances, masks_root: Path) -> None:
    frame_dirs = sorted(path for path in masks_root.iterdir() if (path / "instances.json").exists())
    metadata: dict[tuple[int, int], tuple[str | None, str | None]] = {}
    for frame_index, frame_dir in enumerate(frame_dirs):
        payload = json.loads((frame_dir / "instances.json").read_text(encoding="utf-8"))
        for instance in payload.get("instances", []):
            label = instance.get("label")
            metadata[(frame_index, int(instance["instance_id"]))] = (
                str(label) if label is not None else None,
                infer_label_family(str(label) if label is not None else None),
            )
    for local_instance in local_instances:
        label, label_family = metadata.get(
            (local_instance.frame_index, local_instance.instance_id),
            (None, None),
        )
        local_instance.label = label
        local_instance.label_family = label_family


def main() -> int:
    args = parse_args()
    arrays = load_vote_evidence(args.vote_evidence)
    local_instances = aggregate_local_instances(*arrays)
    if args.masks_root is not None:
        _attach_label_metadata(local_instances, args.masks_root)
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
        require_same_label_family=args.require_same_label_family,
    )
    global_object_ids = assign_global_objects(
        candidate_instances,
        pairs,
        total_local_count=len(local_instances),
    )

    save_association_manifest(
        local_instances,
        global_object_ids,
        pairs,
        args.output_root / "association_manifest.json",
    )
    save_association_edges(pairs, args.output_root / "association_edges.npz")
    print(
        "Associated "
        f"{len(local_instances)} local instances into "
        f"{int((set(global_object_ids.tolist()) - {-1}).__len__()) if len(global_object_ids) else 0} global objects "
        f"({len(candidate_instances)} candidates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
