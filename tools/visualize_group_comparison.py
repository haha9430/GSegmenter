"""Colorize a Gaussian PLY by GT/pred grouping agreement categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import (
    GAUSSIAN_COMPARISON_LABELS,
    classify_gaussian_grouping,
    evaluate_gaussian_grouping,
    load_gaussian_table,
    rgb_to_sh_dc,
    summarize_gaussian_comparison,
    write_gaussian_table,
)


_CATEGORY_RGB = {
    0: np.asarray([0.55, 0.55, 0.55], dtype=np.float32),  # background
    1: np.asarray([0.16, 0.82, 0.28], dtype=np.float32),  # matched
    2: np.asarray([0.95, 0.22, 0.22], dtype=np.float32),  # gt only / missed
    3: np.asarray([0.16, 0.42, 0.95], dtype=np.float32),  # pred only / false positive
    4: np.asarray([0.98, 0.70, 0.15], dtype=np.float32),  # conflict / wrong match
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Gaussian PLY colored by GT/pred grouping agreement."
    )
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--gt-object-ids", type=Path, required=True)
    parser.add_argument("--pred-object-ids", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument(
        "--background-opacity-scale",
        type=float,
        default=0.35,
        help="Opacity scale applied to background (category 0) gaussians to reduce clutter.",
    )
    parser.add_argument(
        "--flatten-selected-sh",
        action="store_true",
        help="Zero higher-order SH channels so comparison colors read clearly in viewers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table, header_properties = load_gaussian_table(args.ply_path)
    gt_object_ids = np.load(args.gt_object_ids)
    pred_object_ids = np.load(args.pred_object_ids)
    if table.shape[0] != gt_object_ids.shape[0] or table.shape[0] != pred_object_ids.shape[0]:
        raise ValueError(
            "PLY row count, GT object id count, and predicted object id count must match."
        )
    for key in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if key not in table.dtype.names:
            raise ValueError(f"Input PLY is missing required SH DC channel {key!r}.")

    metrics = evaluate_gaussian_grouping(gt_object_ids, pred_object_ids, min_iou=float(args.min_iou))
    categories = classify_gaussian_grouping(gt_object_ids, pred_object_ids, metrics.matches)
    category_counts = summarize_gaussian_comparison(categories)

    highlighted = table.copy()
    dc = np.stack(
        [highlighted["f_dc_0"], highlighted["f_dc_1"], highlighted["f_dc_2"]],
        axis=1,
    ).astype(np.float32)
    for category_code, rgb in _CATEGORY_RGB.items():
        mask = categories == np.int8(category_code)
        if not np.any(mask):
            continue
        dc[mask] = rgb_to_sh_dc(rgb)[None, :]
    highlighted["f_dc_0"] = dc[:, 0]
    highlighted["f_dc_1"] = dc[:, 1]
    highlighted["f_dc_2"] = dc[:, 2]

    if args.flatten_selected_sh:
        for property_name in highlighted.dtype.names:
            if property_name.startswith("f_rest_"):
                highlighted[property_name] = np.float32(0.0)

    if "opacity" in highlighted.dtype.names:
        background_mask = categories == np.int8(0)
        if np.any(background_mask):
            opacity_scale = float(args.background_opacity_scale)
            if opacity_scale <= 0.0:
                raise ValueError("--background-opacity-scale must be positive.")
            highlighted["opacity"][background_mask] = highlighted["opacity"][background_mask] + np.float32(
                np.log(opacity_scale)
            )

    write_gaussian_table(args.output_path, highlighted, header_properties)

    payload = {
        "total_gaussians": int(categories.shape[0]),
        "category_counts": category_counts,
        "category_labels": {str(code): label for code, label in GAUSSIAN_COMPARISON_LABELS.items()},
        "metrics": {
            "matched_pair_count": int(metrics.matched_pair_count),
            "mean_matched_iou": float(metrics.mean_matched_iou),
            "gt_group_recall": float(metrics.gt_group_recall),
            "pred_group_precision": float(metrics.pred_group_precision),
            "gt_gaussian_recall": float(metrics.gt_gaussian_recall),
            "pred_gaussian_precision": float(metrics.pred_gaussian_precision),
        },
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote comparison-colored Gaussian PLY to {args.output_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
