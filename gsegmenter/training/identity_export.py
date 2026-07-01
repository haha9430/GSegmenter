"""PLY export helpers for identity-aware Splatfacto checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def build_identity_export_tensors(
    *,
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    shs_0: np.ndarray,
    shs_rest: np.ndarray | None,
    ply_color_mode: str = "sh_coeffs",
) -> OrderedDict[str, np.ndarray]:
    """Build a PLY-compatible tensor table from Gaussian parameter arrays."""

    n = int(positions.shape[0])
    tensors: OrderedDict[str, np.ndarray] = OrderedDict()
    tensors["x"] = positions[:, 0]
    tensors["y"] = positions[:, 1]
    tensors["z"] = positions[:, 2]
    tensors["nx"] = np.zeros(n, dtype=np.float32)
    tensors["ny"] = np.zeros(n, dtype=np.float32)
    tensors["nz"] = np.zeros(n, dtype=np.float32)

    if ply_color_mode == "rgb":
        colors = np.clip(shs_0 * 0.28209479177387814 + 0.5, 0.0, 1.0)
        colors = (colors * 255.0).astype(np.uint8)
        tensors["red"] = colors[:, 0]
        tensors["green"] = colors[:, 1]
        tensors["blue"] = colors[:, 2]
    elif ply_color_mode == "sh_coeffs":
        for i in range(shs_0.shape[1]):
            tensors[f"f_dc_{i}"] = shs_0[:, i : i + 1]
        if shs_rest is not None:
            flattened_rest = shs_rest.reshape((n, -1))
            for i in range(flattened_rest.shape[-1]):
                tensors[f"f_rest_{i}"] = flattened_rest[:, i : i + 1]
    else:
        raise ValueError(f"Unsupported ply_color_mode: {ply_color_mode}")

    tensors["opacity"] = opacities
    for i in range(3):
        tensors[f"scale_{i}"] = scales[:, i : i + 1]
    for i in range(4):
        tensors[f"rot_{i}"] = quats[:, i : i + 1]
    return tensors


def filter_finite_and_visible_gaussians(
    tensors: Mapping[str, np.ndarray],
    *,
    opacity_logit_threshold: float = -5.5373,
) -> tuple[OrderedDict[str, np.ndarray], int, int, np.ndarray]:
    """Drop non-finite and near-transparent gaussians before export."""

    ordered = OrderedDict((key, np.asarray(value)) for key, value in tensors.items())
    if not ordered:
        raise ValueError("No tensors were provided for export.")

    count = next(iter(ordered.values())).shape[0]
    select = np.ones(count, dtype=bool)
    invalid_count = 0
    for tensor in ordered.values():
        before = int(select.sum())
        select &= np.isfinite(tensor).all(axis=-1) if tensor.ndim > 1 else np.isfinite(tensor)
        invalid_count += before - int(select.sum())

    opacity = ordered["opacity"].squeeze(-1) if ordered["opacity"].ndim > 1 else ordered["opacity"]
    low_opacity = opacity < opacity_logit_threshold
    low_opacity_count = int(low_opacity.sum())
    select[low_opacity] = False

    filtered = OrderedDict((key, value[select]) for key, value in ordered.items())
    return filtered, invalid_count, low_opacity_count, select


def export_identity_sidecar(
    output_dir: Path,
    *,
    identity_embeddings: torch.Tensor | None,
    keep_mask: np.ndarray | None = None,
) -> Path | None:
    """Write identity embeddings to a sidecar `.npy` for later analysis."""

    if identity_embeddings is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "identity_embeddings.npy"
    array = identity_embeddings.detach().cpu().numpy()
    if keep_mask is not None:
        if array.shape[0] != keep_mask.shape[0]:
            raise ValueError(
                f"Identity sidecar length {array.shape[0]} does not match export mask {keep_mask.shape[0]}"
            )
        array = array[keep_mask]
    np.save(path, array)
    return path


def classify_gaussian_identities(
    *,
    identity_embeddings: torch.Tensor,
    classifier: torch.nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify per-Gaussian identity embeddings with the learned 1x1 head.

    Args:
        identity_embeddings: `(N, D)` learned Gaussian identity features.
        classifier: `Conv2d(D, C, 1)` module from the identity field.

    Returns:
        `(identity_ids, probabilities)` where `identity_ids` has shape `(N,)`
        and `probabilities` has shape `(N, C)`.
    """

    if identity_embeddings.ndim != 2:
        raise ValueError(
            f"Expected identity embeddings with shape `(N, D)`, got {tuple(identity_embeddings.shape)}"
        )
    classifier.eval()
    with torch.no_grad():
        pixel_like = identity_embeddings.T.unsqueeze(0).unsqueeze(-1)
        logits = classifier(pixel_like).squeeze(0).squeeze(-1).T
        probabilities = torch.softmax(logits, dim=-1)
        identity_ids = torch.argmax(probabilities, dim=-1)
    return (
        identity_ids.detach().cpu().numpy().astype(np.int32),
        probabilities.detach().cpu().numpy().astype(np.float32),
    )


def identity_palette_rgb(num_classes: int) -> np.ndarray:
    """Return deterministic high-contrast RGB colors in `[0, 1]`."""

    base = np.asarray(
        [
            [0.90, 0.18, 0.18],
            [0.15, 0.70, 0.25],
            [0.20, 0.42, 0.95],
            [0.98, 0.68, 0.15],
            [0.72, 0.22, 0.88],
            [0.08, 0.72, 0.72],
            [0.95, 0.42, 0.62],
            [0.58, 0.80, 0.16],
        ],
        dtype=np.float32,
    )
    if num_classes <= base.shape[0]:
        return base[:num_classes]

    colors = [base[index % base.shape[0]].copy() for index in range(num_classes)]
    for index in range(base.shape[0], num_classes):
        hue = (index * 0.61803398875) % 1.0
        sector = int(hue * 6.0)
        frac = hue * 6.0 - sector
        value = 0.95
        chroma = 0.70
        x = chroma * (1.0 - abs(frac % 2.0 - 1.0))
        rgb = [
            (chroma, x, 0.0),
            (x, chroma, 0.0),
            (0.0, chroma, x),
            (0.0, x, chroma),
            (x, 0.0, chroma),
            (chroma, 0.0, x),
        ][sector % 6]
        colors[index] = np.asarray(rgb, dtype=np.float32) + (value - chroma)
    return np.asarray(colors, dtype=np.float32)


def apply_identity_colors_to_tensors(
    tensors: OrderedDict[str, np.ndarray],
    identity_ids: np.ndarray,
    *,
    palette_rgb: np.ndarray | None = None,
) -> OrderedDict[str, np.ndarray]:
    """Return a copy of Gaussian export tensors with identity-colored SH DC terms."""

    if "f_dc_0" not in tensors or "f_dc_1" not in tensors or "f_dc_2" not in tensors:
        raise ValueError("Identity coloring requires SH coefficient tensors f_dc_0..2.")
    count = next(iter(tensors.values())).shape[0]
    identity_ids = np.asarray(identity_ids, dtype=np.int32)
    if identity_ids.shape != (count,):
        raise ValueError(f"identity_ids must have shape ({count},), got {identity_ids.shape}")
    class_count = int(identity_ids.max() + 1) if identity_ids.size else 0
    if palette_rgb is None:
        palette_rgb = identity_palette_rgb(class_count)
    if palette_rgb.shape[0] < class_count:
        raise ValueError("palette_rgb does not contain enough class colors.")

    colored = OrderedDict((key, value.copy()) for key, value in tensors.items())
    rgb = palette_rgb[identity_ids]
    sh_dc = (rgb - 0.5) / 0.28209479177387814
    colored["f_dc_0"] = sh_dc[:, 0:1].astype(np.float32)
    colored["f_dc_1"] = sh_dc[:, 1:2].astype(np.float32)
    colored["f_dc_2"] = sh_dc[:, 2:3].astype(np.float32)
    for key in colored:
        if key.startswith("f_rest_"):
            colored[key] = np.zeros_like(colored[key], dtype=np.float32)
    return colored
