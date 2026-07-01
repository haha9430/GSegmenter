"""Instance proposals derived from trained Gaussian identity classes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gsegmenter.mapping.cluster_proposals import (
    build_voxel_connected_components,
    filter_and_remap_components,
    summarize_cluster_proposals,
)
from gsegmenter.mapping.gaussian_io import rgb_to_sh_dc, write_gaussian_table


INSTANCE_PALETTE_RGB = np.asarray(
    [
        [0.95, 0.20, 0.20],
        [0.10, 0.78, 0.78],
        [0.18, 0.50, 0.95],
        [0.98, 0.70, 0.15],
        [0.75, 0.22, 0.88],
        [0.20, 0.82, 0.30],
        [0.95, 0.45, 0.65],
        [0.55, 0.85, 0.18],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class IdentityInstanceProposal:
    """One spatial component inside a semantic identity class.

    `instance_id` is scene-global and aligned with the exported Gaussian PLY row
    order. Coordinates are in the exported Gaussian world frame.
    """

    instance_id: int
    class_id: int
    class_name: str
    rank_in_class: int
    gaussian_count: int
    voxel_count: int
    confidence_mean: float
    confidence_min: float
    confidence_max: float
    centroid_xyz: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray


def build_identity_instance_ids(
    *,
    xyz: np.ndarray,
    identity_ids: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    include_class_ids: set[int],
    min_confidence: float,
    voxel_size: float,
    min_voxel_count: int = 1,
    min_gaussians: int = 100,
    max_instances_per_class: int = 0,
) -> tuple[np.ndarray, list[IdentityInstanceProposal]]:
    """Split high-confidence Gaussians in each class into 3D components.

    Args:
        xyz: Gaussian centers with shape `(N, 3)` in exported world coordinates.
        identity_ids: Predicted identity class id per Gaussian, shape `(N,)`.
        probabilities: Class probabilities per Gaussian, shape `(N, C)`.
        class_names: Human-readable class names for class ids.
        include_class_ids: Semantic classes to split into object candidates.
        min_confidence: Minimum probability for a Gaussian to participate.
        voxel_size: Cubic voxel edge length used for 26-connected components.
        min_voxel_count: Drop occupied voxels below this local support count.
        min_gaussians: Drop components smaller than this Gaussian count.
        max_instances_per_class: Keep only the largest K components per class.
            `0` means keep all passing components.

    Returns:
        A dense `(N,)` int32 array where `-1` means no instance, plus proposal
        summaries in global instance id order.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    identity_ids = np.asarray(identity_ids, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if identity_ids.shape != (xyz.shape[0],):
        raise ValueError(f"identity_ids must have shape ({xyz.shape[0]},), got {identity_ids.shape}")
    if probabilities.ndim != 2 or probabilities.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"probabilities must have shape (N, C) with N={xyz.shape[0]}, got {probabilities.shape}"
        )
    if len(class_names) < probabilities.shape[1]:
        raise ValueError("class_names must contain at least one name per probability class.")
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be in [0, 1].")
    if max_instances_per_class < 0:
        raise ValueError("max_instances_per_class must be non-negative.")

    confidence = probabilities[np.arange(probabilities.shape[0]), identity_ids]
    instance_ids = np.full((xyz.shape[0],), -1, dtype=np.int32)
    proposals: list[IdentityInstanceProposal] = []
    next_instance_id = 0

    for class_id in sorted(include_class_ids):
        if class_id < 0 or class_id >= probabilities.shape[1]:
            raise ValueError(f"Class id {class_id} is outside probability range.")
        class_mask = (identity_ids == class_id) & (confidence >= float(min_confidence))
        class_indices = np.flatnonzero(class_mask)
        if class_indices.size == 0:
            continue

        raw_components = build_voxel_connected_components(
            xyz[class_indices],
            voxel_size=voxel_size,
            min_voxel_count=min_voxel_count,
        )
        local_components = filter_and_remap_components(raw_components, min_gaussians=min_gaussians)
        if max_instances_per_class > 0:
            keep_local = set(range(max_instances_per_class))
            local_components = np.where(
                np.isin(local_components, np.asarray(sorted(keep_local), dtype=np.int32)),
                local_components,
                -1,
            ).astype(np.int32)

        local_summaries = summarize_cluster_proposals(
            local_components,
            xyz[class_indices],
            voxel_size=voxel_size,
        )
        local_to_global: dict[int, int] = {}
        for proposal in local_summaries:
            local_id = int(proposal.global_object_id)
            global_id = next_instance_id
            next_instance_id += 1
            local_to_global[local_id] = global_id
            member_indices = class_indices[local_components == local_id]
            member_confidence = confidence[member_indices]
            proposals.append(
                IdentityInstanceProposal(
                    instance_id=global_id,
                    class_id=int(class_id),
                    class_name=class_names[class_id],
                    rank_in_class=local_id,
                    gaussian_count=int(proposal.gaussian_count),
                    voxel_count=int(proposal.voxel_count),
                    confidence_mean=float(member_confidence.mean()),
                    confidence_min=float(member_confidence.min()),
                    confidence_max=float(member_confidence.max()),
                    centroid_xyz=proposal.centroid_xyz,
                    bbox_min_xyz=proposal.bbox_min_xyz,
                    bbox_max_xyz=proposal.bbox_max_xyz,
                )
            )

        selected_local = local_components >= 0
        if np.any(selected_local):
            mapped = np.asarray(
                [local_to_global[int(local_id)] for local_id in local_components[selected_local]],
                dtype=np.int32,
            )
            instance_ids[class_indices[selected_local]] = mapped

    return instance_ids, proposals


def save_identity_instance_summary(
    *,
    output_path: Path,
    proposals: list[IdentityInstanceProposal],
    parameters: dict,
) -> None:
    """Write instance proposal summaries to JSON."""

    payload = {
        "instance_count": len(proposals),
        "parameters": parameters,
        "instances": [
            {
                "instance_id": proposal.instance_id,
                "class_id": proposal.class_id,
                "class_name": proposal.class_name,
                "rank_in_class": proposal.rank_in_class,
                "gaussian_count": proposal.gaussian_count,
                "voxel_count": proposal.voxel_count,
                "confidence_mean": proposal.confidence_mean,
                "confidence_min": proposal.confidence_min,
                "confidence_max": proposal.confidence_max,
                "centroid_xyz": proposal.centroid_xyz.astype(float).tolist(),
                "bbox_min_xyz": proposal.bbox_min_xyz.astype(float).tolist(),
                "bbox_max_xyz": proposal.bbox_max_xyz.astype(float).tolist(),
            }
            for proposal in proposals
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_identity_instance_highlight_ply(
    *,
    output_path: Path,
    table: np.ndarray,
    header_properties: list[tuple[str, str]],
    instance_ids: np.ndarray,
    dim_opacity_scale: float = 0.25,
) -> None:
    """Color each accepted instance while preserving Gaussian Splatting channels."""

    for channel in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if channel not in table.dtype.names:
            raise ValueError(f"Input PLY does not contain required channel {channel!r}.")
    if instance_ids.shape != (table.shape[0],):
        raise ValueError(f"instance_ids must have shape ({table.shape[0]},), got {instance_ids.shape}")
    if dim_opacity_scale <= 0.0:
        raise ValueError("dim_opacity_scale must be positive.")

    colored = table.copy()
    selected = instance_ids >= 0
    for instance_id in sorted(int(value) for value in np.unique(instance_ids[selected])):
        rgb = INSTANCE_PALETTE_RGB[instance_id % INSTANCE_PALETTE_RGB.shape[0]]
        sh_dc = rgb_to_sh_dc(rgb).astype(np.float32)
        mask = instance_ids == instance_id
        colored["f_dc_0"][mask] = sh_dc[0]
        colored["f_dc_1"][mask] = sh_dc[1]
        colored["f_dc_2"][mask] = sh_dc[2]
        for property_name in colored.dtype.names:
            if property_name.startswith("f_rest_"):
                colored[property_name][mask] = np.float32(0.0)

    if "opacity" in colored.dtype.names:
        colored["opacity"][~selected] = colored["opacity"][~selected] + np.float32(np.log(dim_opacity_scale))

    write_gaussian_table(output_path, colored, header_properties)
