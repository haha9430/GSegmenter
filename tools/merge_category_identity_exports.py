"""Merge category binary identity exports into one inspection PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.gaussian_io import load_gaussian_table, rgb_to_sh_dc, write_gaussian_table


DEFAULT_PALETTE: dict[str, tuple[float, float, float]] = {
    "tv": (0.10, 0.85, 0.25),
    "sofa": (0.95, 0.25, 0.20),
    "chair": (0.20, 0.45, 0.95),
    "table": (0.98, 0.70, 0.15),
    "storage": (0.75, 0.22, 0.88),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate high-confidence Gaussians from separate category identity exports. "
            "This is for visual inspection; separate binary training runs do not share "
            "one Gaussian table."
        )
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--category",
        action="append",
        nargs=3,
        metavar=("NAME", "PLY_PATH", "HIGHLIGHT_MASK_NPY"),
        required=True,
        help="Category name, exported identity PLY path, and gaussian_identity_highlight_mask.npy path.",
    )
    parser.add_argument(
        "--min-selected",
        type=int,
        default=1,
        help="Skip a category if fewer than this many highlighted Gaussians are available.",
    )
    return parser.parse_args()


def _set_category_color(table: np.ndarray, category: str) -> np.ndarray:
    for key in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if key not in table.dtype.names:
            raise ValueError(f"PLY table is missing required Gaussian color channel {key!r}.")
    colored = table.copy()
    rgb = np.asarray(DEFAULT_PALETTE.get(category, (0.95, 0.95, 0.95)), dtype=np.float32)
    sh_dc = rgb_to_sh_dc(rgb).astype(np.float32)
    colored["f_dc_0"] = sh_dc[0]
    colored["f_dc_1"] = sh_dc[1]
    colored["f_dc_2"] = sh_dc[2]
    for property_name in colored.dtype.names:
        if property_name.startswith("f_rest_"):
            colored[property_name] = np.float32(0.0)
    return colored


def main() -> int:
    args = parse_args()
    merged_tables: list[np.ndarray] = []
    header_properties: list[tuple[str, str]] | None = None
    summary = {
        "note": (
            "Merged from separate binary identity training runs. Use for inspection, "
            "not as a single jointly-trained semantic Gaussian table."
        ),
        "categories": [],
    }

    for category, ply_path_raw, mask_path_raw in args.category:
        ply_path = Path(ply_path_raw)
        mask_path = Path(mask_path_raw)
        table, current_header = load_gaussian_table(ply_path)
        if header_properties is None:
            header_properties = current_header
        elif current_header != header_properties:
            raise ValueError(f"PLY schema mismatch for category {category!r}: {ply_path}")

        highlight_mask = np.load(mask_path).astype(bool)
        if highlight_mask.shape[0] != table.shape[0]:
            raise ValueError(
                f"Mask length {highlight_mask.shape[0]} does not match PLY row count {table.shape[0]} for {category!r}"
            )
        selected_count = int(np.count_nonzero(highlight_mask))
        if selected_count < args.min_selected:
            summary["categories"].append(
                {
                    "name": category,
                    "ply_path": str(ply_path),
                    "highlight_mask": str(mask_path),
                    "selected_count": selected_count,
                    "merged": False,
                }
            )
            continue
        selected_table = _set_category_color(table[highlight_mask], category)
        merged_tables.append(selected_table)
        summary["categories"].append(
            {
                "name": category,
                "ply_path": str(ply_path),
                "highlight_mask": str(mask_path),
                "selected_count": selected_count,
                "merged": True,
                "rgb": list(DEFAULT_PALETTE.get(category, (0.95, 0.95, 0.95))),
            }
        )

    if header_properties is None or not merged_tables:
        raise ValueError("No category Gaussians were available to merge.")

    merged = np.concatenate(merged_tables)
    write_gaussian_table(args.output_path, merged, header_properties)
    summary["merged_gaussian_count"] = int(merged.shape[0])
    summary["output_path"] = str(args.output_path)
    summary_path = args.summary_path or args.output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote merged category PLY to {args.output_path}")
    print(f"Merged {merged.shape[0]} gaussians across {len(merged_tables)} categories")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
