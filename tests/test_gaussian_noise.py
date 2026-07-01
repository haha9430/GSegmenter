from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.mapping.gaussian_io import GaussianCloud
from gsegmenter.mapping.gaussian_noise import (
    GaussianPruneSpec,
    GaussianQualitySpec,
    build_gaussian_prune_mask,
    build_gaussian_quality_scores,
    compute_gaussian_noise_report,
    compute_isolated_mask,
    filter_gaussian_sidecar,
)


def _make_cloud() -> GaussianCloud:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.0],
            [0.1, 0.0, 0.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    return GaussianCloud(
        vertex_count=4,
        properties={
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "opacity": np.array([[0.0], [0.0], [0.0], [-10.0]], dtype=np.float32),
            "scale_0": np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
            "scale_1": np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
            "scale_2": np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
        },
    )


def test_compute_isolated_mask_marks_sparse_voxel() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [0.02, 0.01, 0.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    isolated = compute_isolated_mask(xyz, voxel_size=0.1, min_neighbor_count=2)
    assert isolated.tolist() == [False, False, True]


def test_build_gaussian_prune_mask_filters_low_opacity_and_isolated() -> None:
    cloud = _make_cloud()
    spec = GaussianPruneSpec(
        opacity_logit_threshold=-5.0,
        voxel_size=0.1,
        isolated_min_neighbor_count=2,
        remove_isolated=True,
        remove_low_opacity=True,
        remove_extreme_scales=False,
        remove_radius_outliers=False,
    )
    keep = build_gaussian_prune_mask(cloud, spec)
    assert keep.tolist() == [True, True, True, False]


def test_compute_gaussian_noise_report_counts_noise_modes() -> None:
    cloud = _make_cloud()
    report = compute_gaussian_noise_report(
        cloud,
        opacity_logit_threshold=-5.0,
        scale_norm_percentile=75.0,
        voxel_size=0.1,
        isolated_min_neighbor_count=2,
    )
    assert report.gaussian_count == 4
    assert report.low_opacity_count == 1
    assert report.isolated_count >= 1
    assert report.extreme_scale_count >= 1


def test_build_gaussian_quality_scores_downweights_noise_modes() -> None:
    cloud = _make_cloud()
    spec = GaussianQualitySpec(
        opacity_alpha_floor=0.05,
        scale_norm_percentile=75.0,
        voxel_size=0.1,
        isolated_min_neighbor_count=2,
    )

    quality, diagnostics = build_gaussian_quality_scores(cloud, spec)

    assert quality.shape == (4,)
    assert diagnostics["isolated"].shape == (4,)
    assert quality[:3].min() > 0.0
    assert quality[3] == 0.0
    assert diagnostics["low_opacity"][3]


def test_filter_gaussian_sidecar_applies_same_mask(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "identity.npy"
    output_path = tmp_path / "identity_filtered.npy"
    np.save(sidecar_path, np.arange(12, dtype=np.float32).reshape(4, 3))
    keep = np.array([True, False, True, False], dtype=bool)

    filter_gaussian_sidecar(sidecar_path, keep, output_path)

    filtered = np.load(output_path)
    assert filtered.shape == (2, 3)
    assert filtered[1, 0] == 6.0


def test_compute_gaussian_noise_report_handles_empty_cloud() -> None:
    cloud = GaussianCloud(
        vertex_count=0,
        properties={
            "x": np.zeros((0,), dtype=np.float32),
            "y": np.zeros((0,), dtype=np.float32),
            "z": np.zeros((0,), dtype=np.float32),
        },
    )

    report = compute_gaussian_noise_report(cloud)

    assert report.gaussian_count == 0
    assert report.low_opacity_count == 0
    assert report.isolated_count == 0
    assert report.extreme_scale_count == 0
