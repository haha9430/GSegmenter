"""Highlight Gaussians voted by one frame-local mask instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping import load_gaussian_table, rgb_to_sh_dc, write_gaussian_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a 3DGS PLY highlighting one local mask vote.")
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--vote-evidence", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--frame-stem", type=str, default=None)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--masks-root", type=Path, default=None)
    parser.add_argument("--highlight-rgb", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--dim-opacity-scale", type=float, default=0.25)
    parser.add_argument("--selected-opacity-scale", type=float, default=1.5)
    parser.add_argument("--flatten-selected-sh", action="store_true")
    return parser.parse_args()


def _resolve_frame_index(frame_index: int | None, frame_stem: str | None, masks_root: Path | None) -> int:
    if frame_index is not None:
        return int(frame_index)
    if frame_stem is None or masks_root is None:
        raise ValueError("Use --frame-index, or provide both --frame-stem and --masks-root.")
    frame_dirs = [path for path in sorted(masks_root.iterdir()) if (path / "instances.json").exists()]
    for index, frame_dir in enumerate(frame_dirs):
        if frame_dir.name == frame_stem:
            return index
    raise ValueError(f"Could not resolve frame stem {frame_stem!r} under {masks_root}")


def _load_instance_label(masks_root: Path | None, frame_stem: str | None, instance_id: int) -> str | None:
    if masks_root is None or frame_stem is None:
        return None
    manifest_path = masks_root / frame_stem / "instances.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for instance in payload.get("instances", []):
        if int(instance["instance_id"]) == int(instance_id):
            return str(instance.get("label", ""))
    return None


def main() -> int:
    args = parse_args()
    resolved_frame_index = _resolve_frame_index(args.frame_index, args.frame_stem, args.masks_root)
    table, header_properties = load_gaussian_table(args.ply_path)
    evidence = np.load(args.vote_evidence)
    frame_indices = evidence["frame_indices"]
    instance_ids = evidence["instance_ids"]
    gaussian_indices = evidence["gaussian_indices"]
    weights = evidence["weights"]

    vote_mask = (frame_indices == resolved_frame_index) & (instance_ids == int(args.instance_id))
    selected_indices = np.unique(gaussian_indices[vote_mask].astype(np.int64))
    if selected_indices.size == 0:
        raise ValueError(
            f"No vote evidence found for frame_index={resolved_frame_index}, instance_id={args.instance_id}"
        )
    if selected_indices.max(initial=-1) >= table.shape[0]:
        raise ValueError("Vote evidence contains Gaussian indices beyond the input PLY vertex count.")

    selected_mask = np.zeros((table.shape[0],), dtype=bool)
    selected_mask[selected_indices] = True
    highlighted = table.copy()
    target_dc = rgb_to_sh_dc(np.asarray(args.highlight_rgb, dtype=np.float32)).astype(np.float32)
    for channel, value in zip(("f_dc_0", "f_dc_1", "f_dc_2"), target_dc, strict=True):
        if channel not in highlighted.dtype.names:
            raise ValueError(f"Input PLY does not contain required color channel {channel!r}.")
        highlighted[channel][selected_mask] = value
    if args.flatten_selected_sh:
        for property_name in highlighted.dtype.names:
            if property_name.startswith("f_rest_"):
                highlighted[property_name][selected_mask] = np.float32(0.0)
    if "opacity" in highlighted.dtype.names:
        if args.dim_opacity_scale <= 0.0 or args.selected_opacity_scale <= 0.0:
            raise ValueError("Opacity scales must be positive.")
        highlighted["opacity"][~selected_mask] = highlighted["opacity"][~selected_mask] + np.float32(
            np.log(float(args.dim_opacity_scale))
        )
        highlighted["opacity"][selected_mask] = highlighted["opacity"][selected_mask] + np.float32(
            np.log(float(args.selected_opacity_scale))
        )

    write_gaussian_table(args.output_path, highlighted, header_properties)
    label = _load_instance_label(args.masks_root, args.frame_stem, int(args.instance_id))
    selected_weight = float(weights[vote_mask].sum()) if np.any(vote_mask) else 0.0
    print(
        f"Wrote {args.output_path} with {selected_indices.size} highlighted gaussians "
        f"for frame_index={resolved_frame_index}, instance_id={args.instance_id}, "
        f"label={label!r}, vote_weight={selected_weight:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
