from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.editor.repair import cleanup_source_region_appearance  # noqa: E402


def _make_table() -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("f_rest_0", "<f4"),
        ]
    )
    table = np.zeros((4,), dtype=dtype)
    table["x"] = np.array([0.0, 0.08, 0.16, 1.0], dtype=np.float32)
    table["y"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    table["z"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    table["f_dc_0"] = np.array([0.8, -0.8, 0.2, 0.5], dtype=np.float32)
    table["f_dc_1"] = np.array([0.8, -0.8, 0.2, 0.5], dtype=np.float32)
    table["f_dc_2"] = np.array([0.8, -0.8, 0.2, 0.5], dtype=np.float32)
    table["opacity"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    table["f_rest_0"] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return table


def test_cleanup_source_region_blends_inner_colors_and_zeroes_sh() -> None:
    table = _make_table()
    object_ids = np.array([148, -1, -1, -1], dtype=np.int32)
    repaired, summary = cleanup_source_region_appearance(
        table,
        object_ids,
        target_object_id=148,
        source_bbox_min_xyz=np.array([-0.05, -0.05, -0.05], dtype=np.float32),
        source_bbox_max_xyz=np.array([0.12, 0.05, 0.05], dtype=np.float32),
        shell_margin=0.1,
        color_blend=1.0,
        opacity_scale=0.5,
    )

    assert summary.inner_count == 1
    assert summary.shell_count == 1
    assert np.isclose(float(repaired["f_dc_0"][1]), 0.2)
    assert np.isclose(float(repaired["f_dc_1"][1]), 0.2)
    assert np.isclose(float(repaired["f_dc_2"][1]), 0.2)
    assert np.isclose(float(repaired["f_rest_0"][1]), 0.0)
    assert np.isclose(float(repaired["opacity"][1]), np.log(0.5), atol=1e-6)


def test_cleanup_source_region_is_noop_without_shell_points() -> None:
    table = _make_table()[:2].copy()
    object_ids = np.array([148, -1], dtype=np.int32)
    repaired, summary = cleanup_source_region_appearance(
        table,
        object_ids,
        target_object_id=148,
        source_bbox_min_xyz=np.array([-0.05, -0.05, -0.05], dtype=np.float32),
        source_bbox_max_xyz=np.array([0.12, 0.05, 0.05], dtype=np.float32),
        shell_margin=0.01,
    )

    assert summary.shell_count == 0
    assert np.allclose(repaired["f_dc_0"], table["f_dc_0"])


def test_cleanup_source_region_opacity_only_preserves_color() -> None:
    table = _make_table()
    object_ids = np.array([148, -1, -1, -1], dtype=np.int32)
    repaired, summary = cleanup_source_region_appearance(
        table,
        object_ids,
        target_object_id=148,
        source_bbox_min_xyz=np.array([-0.05, -0.05, -0.05], dtype=np.float32),
        source_bbox_max_xyz=np.array([0.12, 0.05, 0.05], dtype=np.float32),
        shell_margin=0.1,
        color_blend=1.0,
        opacity_scale=0.5,
        mode="opacity_only",
        zero_high_order_sh=True,
    )

    assert summary.inner_count == 1
    assert np.isclose(float(repaired["f_dc_0"][1]), float(table["f_dc_0"][1]))
    assert np.isclose(float(repaired["f_dc_1"][1]), float(table["f_dc_1"][1]))
    assert np.isclose(float(repaired["f_dc_2"][1]), float(table["f_dc_2"][1]))
    assert np.isclose(float(repaired["f_rest_0"][1]), float(table["f_rest_0"][1]))
    assert np.isclose(float(repaired["opacity"][1]), np.log(0.5), atol=1e-6)
