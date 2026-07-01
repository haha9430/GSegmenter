"""Create a highlighted Gaussian PLY while preserving original 3DGS colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.gaussian_io import load_gaussian_table, rgb_to_sh_dc, write_gaussian_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a copy of splat.ply with selected Gaussian groups highlighted."
    )
    parser.add_argument("--ply-path", type=Path, required=True)
    parser.add_argument("--object-ids", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, default=None)
    parser.add_argument("--skip-largest-n", type=int, default=0)
    parser.add_argument("--min-group-size", type=int, default=0)
    parser.add_argument(
        "--include-labels",
        type=str,
        nargs="*",
        default=None,
        help="Keep only groups whose label matches one of these values in groups_json.",
    )
    parser.add_argument("--include-object-ids", type=int, nargs="*", default=None)
    parser.add_argument("--exclude-object-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--multi-color",
        action="store_true",
        help="Assign a distinct deterministic color per selected object id.",
    )
    parser.add_argument(
        "--highlight-rgb",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(1.0, 0.0, 0.0),
        help="Highlight color in normalized RGB space.",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend amount between original color and highlight color. 1.0 means full replacement.",
    )
    parser.add_argument(
        "--dim-opacity-scale",
        type=float,
        default=1.0,
        help="Scale opacity logits of non-selected Gaussians. Values below 1.0 reduce background prominence.",
    )
    parser.add_argument(
        "--selected-opacity-scale",
        type=float,
        default=1.0,
        help="Scale opacity logits of selected Gaussians. Values above 1.0 make highlighted groups denser.",
    )
    parser.add_argument(
        "--flatten-selected-sh",
        action="store_true",
        help="Zero higher-order SH terms for selected Gaussians so highlight colors read clearly in viewers.",
    )
    return parser.parse_args()


def _group_object_id(group: dict) -> int:
    """Read a group id from either legacy or InteriorGS group JSON schemas."""

    if "global_object_id" in group:
        return int(group["global_object_id"])
    if "object_id" in group:
        return int(group["object_id"])
    raise KeyError("Group record does not contain 'global_object_id' or 'object_id'.")


def _load_keep_object_ids(
    groups_json: Path | None,
    skip_largest_n: int,
    min_group_size: int,
    include_labels: list[str] | None,
    include_object_ids: list[int] | None,
    exclude_object_ids: list[int] | None,
) -> set[int] | None:
    keep_object_ids: set[int] | None = None
    if groups_json is not None:
        payload = json.loads(groups_json.read_text(encoding="utf-8"))
        groups = sorted(payload["groups"], key=lambda group: int(group["gaussian_count"]), reverse=True)
        if include_labels is not None:
            allowed_labels = {label.casefold() for label in include_labels}
            groups = [
                group
                for group in groups
                if str(group.get("label", "")).casefold() in allowed_labels
            ]
        skipped = {_group_object_id(group) for group in groups[:skip_largest_n]} if skip_largest_n > 0 else set()
        keep_object_ids = {
            _group_object_id(group)
            for group in groups
            if int(group["gaussian_count"]) >= min_group_size and _group_object_id(group) not in skipped
        }
    if include_object_ids is not None:
        explicit = {int(object_id) for object_id in include_object_ids}
        keep_object_ids = explicit if keep_object_ids is None else (keep_object_ids & explicit)
    if exclude_object_ids is not None and keep_object_ids is not None:
        keep_object_ids -= {int(object_id) for object_id in exclude_object_ids}
    return keep_object_ids


def _select_gaussians(object_ids: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    keep_object_ids = _load_keep_object_ids(
        args.groups_json,
        args.skip_largest_n,
        args.min_group_size,
        args.include_labels,
        args.include_object_ids,
        args.exclude_object_ids,
    )
    if keep_object_ids is None:
        selection = np.ones_like(object_ids, dtype=bool)
        if args.exclude_object_ids is not None:
            selection &= ~np.isin(object_ids, np.asarray(args.exclude_object_ids, dtype=np.int64))
    else:
        selection = np.isin(object_ids, np.asarray(sorted(keep_object_ids), dtype=np.int64))
    return selection


def _palette_rgb(object_id: int) -> np.ndarray:
    """Return a high-visibility deterministic RGB color for an object id."""

    palette = np.asarray(
        [
            [0.95, 0.20, 0.20],
            [0.15, 0.75, 0.25],
            [0.20, 0.45, 0.95],
            [0.98, 0.70, 0.18],
            [0.75, 0.22, 0.85],
            [0.10, 0.78, 0.78],
            [0.95, 0.45, 0.65],
            [0.60, 0.85, 0.18],
        ],
        dtype=np.float32,
    )
    return palette[abs(int(object_id)) % len(palette)]


def main() -> int:
    args = parse_args()
    table, header_properties = load_gaussian_table(args.ply_path)
    object_ids = np.load(args.object_ids)
    if object_ids.shape[0] != table.shape[0]:
        raise ValueError(
            f"Object id count {object_ids.shape[0]} does not match Gaussian count {table.shape[0]}"
        )
    for key in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if key not in table.dtype.names:
            raise ValueError(f"Input PLY does not contain required SH DC channel {key!r}.")

    selection = _select_gaussians(object_ids, args)
    selected_count = int(np.count_nonzero(selection))
    if selected_count == 0:
        raise ValueError("No Gaussians matched the requested group filters.")

    highlight_rgb = np.asarray(args.highlight_rgb, dtype=np.float32)
    if np.any(highlight_rgb < 0.0) or np.any(highlight_rgb > 1.0):
        raise ValueError("--highlight-rgb values must stay in [0, 1].")
    blend = float(args.blend)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("--blend must stay in [0, 1].")
    dim_opacity_scale = float(args.dim_opacity_scale)
    if dim_opacity_scale <= 0.0:
        raise ValueError("--dim-opacity-scale must be positive.")
    selected_opacity_scale = float(args.selected_opacity_scale)
    if selected_opacity_scale <= 0.0:
        raise ValueError("--selected-opacity-scale must be positive.")

    highlighted = table.copy()
    original_dc = np.stack(
        [highlighted["f_dc_0"], highlighted["f_dc_1"], highlighted["f_dc_2"]],
        axis=1,
    ).astype(np.float32)
    if args.multi_color:
        selected_object_ids = sorted({int(object_id) for object_id in object_ids[selection] if int(object_id) >= 0})
        for selected_object_id in selected_object_ids:
            object_mask = object_ids == selected_object_id
            target_dc = rgb_to_sh_dc(_palette_rgb(selected_object_id)).astype(np.float32)
            original_dc[object_mask] = (1.0 - blend) * original_dc[object_mask] + blend * target_dc[None, :]
    else:
        target_dc = rgb_to_sh_dc(highlight_rgb).astype(np.float32)
        original_dc[selection] = (1.0 - blend) * original_dc[selection] + blend * target_dc[None, :]
    highlighted["f_dc_0"] = original_dc[:, 0]
    highlighted["f_dc_1"] = original_dc[:, 1]
    highlighted["f_dc_2"] = original_dc[:, 2]
    if args.flatten_selected_sh:
        for property_name in highlighted.dtype.names:
            if property_name.startswith("f_rest_"):
                highlighted[property_name][selection] = np.float32(0.0)
    if "opacity" in highlighted.dtype.names:
        non_selected = ~selection
        if dim_opacity_scale != 1.0:
            highlighted["opacity"][non_selected] = highlighted["opacity"][non_selected] + np.float32(
                np.log(dim_opacity_scale)
            )
        if selected_opacity_scale != 1.0:
            highlighted["opacity"][selection] = highlighted["opacity"][selection] + np.float32(
                np.log(selected_opacity_scale)
            )

    write_gaussian_table(args.output_path, highlighted, header_properties)
    print(
        f"Wrote highlighted Gaussian PLY to {args.output_path} "
        f"with {selected_count} highlighted gaussians"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
