"""Evaluate predicted Gaussian object ids against a ground-truth grouping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import evaluate_gaussian_grouping, metrics_to_json_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare predicted Gaussian groups to GT Gaussian groups."
    )
    parser.add_argument("--gt-object-ids", type=Path, required=True, help="Ground-truth gaussian_object_ids.npy")
    parser.add_argument("--pred-object-ids", type=Path, required=True, help="Predicted gaussian_object_ids.npy")
    parser.add_argument("--gt-groups-json", type=Path, default=None, help="Optional GT gaussian_groups.json for per-label metrics.")
    parser.add_argument("--min-iou", type=float, default=0.1, help="Minimum IoU required for one-to-one group matching.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to persist the evaluation summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gt_object_ids = np.load(args.gt_object_ids)
    pred_object_ids = np.load(args.pred_object_ids)
    metrics = evaluate_gaussian_grouping(
        gt_object_ids,
        pred_object_ids,
        min_iou=float(args.min_iou),
        gt_groups_json=args.gt_groups_json,
    )
    payload = metrics_to_json_dict(metrics)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
