"""Diagnostics and pruning helpers for noisy Gaussian clouds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gsegmenter.mapping.gaussian_io import GaussianCloud


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def infer_voxel_size(cloud: GaussianCloud) -> float:
    """Infer a scene-relative voxel size for sparse outlier detection."""

    xyz = cloud.xyz
    if xyz.shape[0] == 0:
        return 1e-3
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    if cloud.scales is not None:
        linear_scales = np.exp(cloud.scales)
        median_scale = float(np.median(linear_scales))
    else:
        median_scale = 0.0
    return max(diagonal / 256.0, median_scale * 4.0, 1e-3)


def compute_isolated_mask(
    xyz: np.ndarray,
    *,
    voxel_size: float,
    min_neighbor_count: int = 2,
) -> np.ndarray:
    """Mark gaussians that live in sparse voxels with too few local neighbors."""

    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    voxel_coords = np.floor(xyz / float(voxel_size)).astype(np.int32)
    unique_voxels, inverse, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    occupancy = {tuple(coord.tolist()): int(count) for coord, count in zip(unique_voxels, counts)}

    isolated = np.zeros((xyz.shape[0],), dtype=bool)
    for index, coord in enumerate(voxel_coords):
        total_neighbors = 0
        cx, cy, cz = (int(coord[0]), int(coord[1]), int(coord[2]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    total_neighbors += occupancy.get((cx + dx, cy + dy, cz + dz), 0)
        if total_neighbors < int(min_neighbor_count):
            isolated[index] = True
    return isolated


@dataclass(slots=True)
class GaussianNoiseReport:
    """Compact diagnostics for a Gaussian cloud before identity grouping."""

    gaussian_count: int
    voxel_size: float
    low_opacity_count: int
    isolated_count: int
    extreme_scale_count: int
    opacity_quantiles: dict[str, float]
    scale_norm_quantiles: dict[str, float]
    radius_quantiles: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class GaussianPruneSpec:
    """Configurable pruning thresholds for noisy gaussian removal."""

    opacity_logit_threshold: float = -5.0
    scale_norm_percentile: float = 99.5
    radius_percentile: float = 99.9
    radius_multiplier: float = 1.0
    voxel_size: float | None = None
    isolated_min_neighbor_count: int = 2
    remove_isolated: bool = True
    remove_low_opacity: bool = True
    remove_extreme_scales: bool = True
    remove_radius_outliers: bool = False


@dataclass(slots=True)
class GaussianQualitySpec:
    """Thresholds used to score Gaussian reliability before mask lifting.

    The score is intended as a soft lifting weight, not as a rendering metric.
    It favors opaque, locally supported Gaussians with non-extreme scale.
    """

    opacity_alpha_floor: float = 0.05
    scale_norm_percentile: float = 99.5
    radius_percentile: float = 99.9
    voxel_size: float | None = None
    isolated_min_neighbor_count: int = 2
    isolated_quality: float = 0.0
    extreme_scale_quality: float = 0.25
    radius_outlier_quality: float = 0.25


def compute_gaussian_noise_report(
    cloud: GaussianCloud,
    *,
    opacity_logit_threshold: float = -5.0,
    scale_norm_percentile: float = 99.5,
    voxel_size: float | None = None,
    isolated_min_neighbor_count: int = 2,
) -> GaussianNoiseReport:
    """Summarize likely noise modes in an exported Gaussian cloud."""

    xyz = cloud.xyz
    if xyz.shape[0] == 0:
        zero_quantiles = {"p50": 0.0, "p90": 0.0, "p99": 0.0, "p99_9": 0.0}
        return GaussianNoiseReport(
            gaussian_count=0,
            voxel_size=float(voxel_size if voxel_size is not None else infer_voxel_size(cloud)),
            low_opacity_count=0,
            isolated_count=0,
            extreme_scale_count=0,
            opacity_quantiles=zero_quantiles.copy(),
            scale_norm_quantiles=zero_quantiles.copy(),
            radius_quantiles=zero_quantiles.copy(),
        )

    voxel_size = float(voxel_size if voxel_size is not None else infer_voxel_size(cloud))
    opacities = cloud.opacities if cloud.opacities is not None else np.zeros((xyz.shape[0],), dtype=np.float32)
    opacity_alpha = _sigmoid(opacities.reshape(-1))
    radius = np.linalg.norm(xyz - np.median(xyz, axis=0, keepdims=True), axis=1)

    if cloud.scales is not None:
        linear_scales = np.exp(cloud.scales)
        scale_norm = np.linalg.norm(linear_scales, axis=1)
        scale_cutoff = float(np.percentile(scale_norm, float(scale_norm_percentile)))
        extreme_scale_count = int((scale_norm > scale_cutoff).sum())
    else:
        scale_norm = np.zeros((xyz.shape[0],), dtype=np.float32)
        extreme_scale_count = 0

    isolated = compute_isolated_mask(
        xyz,
        voxel_size=voxel_size,
        min_neighbor_count=isolated_min_neighbor_count,
    )

    def qmap(values: np.ndarray) -> dict[str, float]:
        return {
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
            "p99_9": float(np.percentile(values, 99.9)),
        }

    return GaussianNoiseReport(
        gaussian_count=int(xyz.shape[0]),
        voxel_size=voxel_size,
        low_opacity_count=int((opacities.reshape(-1) < float(opacity_logit_threshold)).sum()),
        isolated_count=int(isolated.sum()),
        extreme_scale_count=extreme_scale_count,
        opacity_quantiles=qmap(opacity_alpha),
        scale_norm_quantiles=qmap(scale_norm),
        radius_quantiles=qmap(radius),
    )


def build_gaussian_quality_scores(
    cloud: GaussianCloud,
    spec: GaussianQualitySpec,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build per-Gaussian quality scores and diagnostic masks.

    Args:
        cloud: Exported Gaussian table. Expected arrays are shaped `(N,)` for
            opacity and `(N, 3)` for centers/scales.
        spec: Quality scoring thresholds.

    Returns:
        A `(N,)` float32 quality array in `[0, 1]` and boolean diagnostic masks.
        These scores should be applied outside hot rendering paths, before
        multi-view SAM vote lifting.
    """

    xyz = cloud.xyz
    count = xyz.shape[0]
    if count == 0:
        return np.zeros((0,), dtype=np.float32), {
            "low_opacity": np.zeros((0,), dtype=bool),
            "isolated": np.zeros((0,), dtype=bool),
            "extreme_scale": np.zeros((0,), dtype=bool),
            "radius_outlier": np.zeros((0,), dtype=bool),
        }

    if cloud.opacities is not None:
        opacity_alpha = _sigmoid(cloud.opacities.reshape(-1))
    else:
        opacity_alpha = np.ones((count,), dtype=np.float32)
    alpha_floor = max(float(spec.opacity_alpha_floor), 1e-6)
    opacity_quality = np.clip((opacity_alpha - alpha_floor) / (1.0 - alpha_floor), 0.0, 1.0)
    low_opacity = opacity_alpha < alpha_floor

    if cloud.scales is not None:
        linear_scales = np.exp(cloud.scales)
        scale_norm = np.linalg.norm(linear_scales, axis=1)
        scale_cutoff = float(np.percentile(scale_norm, float(spec.scale_norm_percentile)))
        extreme_scale = scale_norm > scale_cutoff
    else:
        extreme_scale = np.zeros((count,), dtype=bool)

    radius = np.linalg.norm(xyz - np.median(xyz, axis=0, keepdims=True), axis=1)
    radius_cutoff = float(np.percentile(radius, float(spec.radius_percentile)))
    radius_outlier = radius > radius_cutoff

    isolated = compute_isolated_mask(
        xyz,
        voxel_size=float(spec.voxel_size if spec.voxel_size is not None else infer_voxel_size(cloud)),
        min_neighbor_count=spec.isolated_min_neighbor_count,
    )

    quality = opacity_quality.astype(np.float32, copy=True)
    quality[isolated] *= float(spec.isolated_quality)
    quality[extreme_scale] *= float(spec.extreme_scale_quality)
    quality[radius_outlier] *= float(spec.radius_outlier_quality)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)
    diagnostics = {
        "low_opacity": low_opacity,
        "isolated": isolated,
        "extreme_scale": extreme_scale,
        "radius_outlier": radius_outlier,
    }
    return quality, diagnostics


def build_gaussian_prune_mask(cloud: GaussianCloud, spec: GaussianPruneSpec) -> np.ndarray:
    """Build a deterministic boolean keep-mask from pruning thresholds."""

    xyz = cloud.xyz
    keep = np.ones((xyz.shape[0],), dtype=bool)

    if spec.remove_low_opacity and cloud.opacities is not None:
        keep &= cloud.opacities.reshape(-1) >= float(spec.opacity_logit_threshold)

    if spec.remove_extreme_scales and cloud.scales is not None:
        linear_scales = np.exp(cloud.scales)
        scale_norm = np.linalg.norm(linear_scales, axis=1)
        scale_cutoff = float(np.percentile(scale_norm, float(spec.scale_norm_percentile)))
        keep &= scale_norm <= scale_cutoff

    if spec.remove_radius_outliers:
        radius = np.linalg.norm(xyz - np.median(xyz, axis=0, keepdims=True), axis=1)
        radius_cutoff = float(np.percentile(radius, float(spec.radius_percentile))) * float(spec.radius_multiplier)
        keep &= radius <= radius_cutoff

    if spec.remove_isolated:
        isolated = compute_isolated_mask(
            xyz,
            voxel_size=float(spec.voxel_size if spec.voxel_size is not None else infer_voxel_size(cloud)),
            min_neighbor_count=spec.isolated_min_neighbor_count,
        )
        keep &= ~isolated

    return keep


def filter_gaussian_sidecar(sidecar_path: Path, keep_mask: np.ndarray, output_path: Path) -> Path:
    """Filter a sidecar `.npy` array with the same Gaussian mask used for the PLY."""

    sidecar = np.load(sidecar_path)
    if sidecar.shape[0] != keep_mask.shape[0]:
        raise ValueError(
            f"Sidecar first dimension {sidecar.shape[0]} does not match gaussian mask {keep_mask.shape[0]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sidecar[keep_mask])
    return output_path
