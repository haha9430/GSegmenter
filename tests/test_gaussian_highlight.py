from __future__ import annotations

import os
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.mapping.gaussian_io import load_gaussian_table, rgb_to_sh_dc, write_gaussian_table
from tools.highlight_group_ply import _load_keep_object_ids


def _write_test_ply(path: Path) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "end_header",
            "",
        ]
    ).encode("ascii")
    row_format = "<7f"
    rows = [
        (0.0, 0.0, 2.0, 0.1, 0.2, 0.3, 0.5),
        (1.0, 0.0, 2.0, -0.1, -0.2, -0.3, 0.7),
    ]
    with path.open("wb") as handle:
        handle.write(header)
        for row in rows:
            handle.write(struct.pack(row_format, *row))


def test_rgb_to_sh_dc_red() -> None:
    coeffs = rgb_to_sh_dc(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    expected = np.array([1.7724538, -1.7724538, -1.7724538], dtype=np.float32)
    assert np.allclose(coeffs, expected, atol=1e-5)


def test_rgb_to_sh_dc_blue() -> None:
    coeffs = rgb_to_sh_dc(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    expected = np.array([-1.7724538, -1.7724538, 1.7724538], dtype=np.float32)
    assert np.allclose(coeffs, expected, atol=1e-5)


def test_gaussian_table_round_trip(tmp_path: Path) -> None:
    input_path = tmp_path / "input.ply"
    output_path = tmp_path / "output.ply"
    _write_test_ply(input_path)

    table, header_properties = load_gaussian_table(input_path)
    table = table.copy()
    table["f_dc_0"][1] = np.float32(1.25)
    write_gaussian_table(output_path, table, header_properties)

    loaded, loaded_properties = load_gaussian_table(output_path)
    assert header_properties == loaded_properties
    assert loaded.shape == table.shape
    assert np.isclose(float(loaded["f_dc_0"][1]), 1.25)
    assert np.isclose(float(loaded["opacity"][0]), 0.5)


def test_load_keep_object_ids_supports_interiorgs_schema_and_labels(tmp_path: Path) -> None:
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(
        """
{
  "groups": [
    {"object_id": 10, "label": "chair", "gaussian_count": 100},
    {"object_id": 11, "label": "table", "gaussian_count": 50},
    {"object_id": 12, "label": "chair", "gaussian_count": 25}
  ]
}
        """.strip(),
        encoding="utf-8",
    )

    keep = _load_keep_object_ids(
        groups_json,
        skip_largest_n=0,
        min_group_size=30,
        include_labels=["chair"],
        include_object_ids=None,
        exclude_object_ids=None,
    )

    assert keep == {10}
