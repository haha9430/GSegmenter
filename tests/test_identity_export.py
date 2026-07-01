from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.training.identity_export import (
    apply_identity_colors_to_tensors,
    build_identity_export_tensors,
    classify_gaussian_identities,
    export_identity_sidecar,
    filter_finite_and_visible_gaussians,
)


def test_build_identity_export_tensors_sh_coeffs() -> None:
    tensors = build_identity_export_tensors(
        positions=np.zeros((2, 3), dtype=np.float32),
        opacities=np.zeros((2, 1), dtype=np.float32),
        scales=np.zeros((2, 3), dtype=np.float32),
        quats=np.zeros((2, 4), dtype=np.float32),
        shs_0=np.ones((2, 3), dtype=np.float32),
        shs_rest=np.ones((2, 2, 3), dtype=np.float32),
        ply_color_mode="sh_coeffs",
    )

    assert "x" in tensors
    assert "f_dc_0" in tensors
    assert "f_rest_0" in tensors
    assert "opacity" in tensors
    assert tensors["rot_3"].shape == (2, 1)


def test_filter_finite_and_visible_gaussians_drops_invalid_and_low_opacity() -> None:
    tensors = {
        "x": np.array([0.0, np.nan, 2.0], dtype=np.float32),
        "opacity": np.array([[0.0], [0.0], [-10.0]], dtype=np.float32),
        "scale_0": np.zeros((3, 1), dtype=np.float32),
    }
    filtered, invalid_count, low_opacity_count, keep_mask = filter_finite_and_visible_gaussians(tensors)

    assert next(iter(filtered.values())).shape[0] == 1
    assert invalid_count >= 1
    assert low_opacity_count == 1
    assert keep_mask.tolist() == [True, False, False]


def test_export_identity_sidecar_writes_npy(tmp_path: Path) -> None:
    path = export_identity_sidecar(
        tmp_path,
        identity_embeddings=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        keep_mask=np.array([True, False, True, False], dtype=bool),
    )

    assert path is not None
    assert path.exists()
    filtered = np.load(path)
    assert filtered.shape == (2, 3)


def test_classify_gaussian_identities_uses_classifier_head() -> None:
    classifier = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]))
    embeddings = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)

    identity_ids, probabilities = classify_gaussian_identities(
        identity_embeddings=embeddings,
        classifier=classifier,
    )

    assert identity_ids.tolist() == [0, 1]
    assert probabilities.shape == (2, 2)


def test_apply_identity_colors_to_tensors_replaces_sh_dc() -> None:
    tensors = build_identity_export_tensors(
        positions=np.zeros((2, 3), dtype=np.float32),
        opacities=np.zeros((2, 1), dtype=np.float32),
        scales=np.zeros((2, 3), dtype=np.float32),
        quats=np.zeros((2, 4), dtype=np.float32),
        shs_0=np.zeros((2, 3), dtype=np.float32),
        shs_rest=np.ones((2, 1, 3), dtype=np.float32),
        ply_color_mode="sh_coeffs",
    )

    colored = apply_identity_colors_to_tensors(
        tensors,
        np.asarray([0, 1], dtype=np.int32),
        palette_rgb=np.asarray([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5]], dtype=np.float32),
    )

    assert not np.allclose(colored["f_dc_0"], tensors["f_dc_0"])
    assert np.allclose(colored["f_rest_0"], 0.0)
