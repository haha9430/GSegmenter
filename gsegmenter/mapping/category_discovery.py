"""Training-free category and instance discovery from lifted 2D masks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from gsegmenter.mapping.cluster_proposals import (
    build_voxel_connected_components,
    filter_and_remap_components,
    summarize_cluster_proposals,
)
from gsegmenter.mapping.identity_instances import write_identity_instance_highlight_ply
from gsegmenter.mapping.lifting import VoteEvidence


@dataclass(slots=True)
class CategoryDiscoverySpec:
    """Regex definition for one first-pass semantic category."""

    name: str
    pattern: re.Pattern[str]


@dataclass(slots=True)
class CategoryInstanceProposal:
    """One spatial instance candidate discovered without identity training."""

    instance_id: int
    category: str
    rank_in_category: int
    gaussian_count: int
    voxel_count: int
    vote_weight_sum: float
    support_count_mean: float
    centroid_xyz: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray


def default_category_specs() -> list[CategoryDiscoverySpec]:
    """Return furniture categories used by the current GSegmenter experiments."""

    raw_specs = [
        ("tv", r"television|tv"),
        ("chair", r"chair|armchair|stool|ottoman"),
        ("table", r"table|desk|nightstand"),
        ("sofa", r"sofa|couch"),
        ("storage", r"cabinet|shelf|bookshelf|wardrobe|drawer|dresser"),
    ]
    return [
        CategoryDiscoverySpec(name=name, pattern=re.compile(pattern, flags=re.IGNORECASE))
        for name, pattern in raw_specs
    ]


def match_category(label: str | None, specs: list[CategoryDiscoverySpec]) -> int | None:
    """Return the first category index whose regex matches a grounded label."""

    if label is None:
        return None
    for index, spec in enumerate(specs):
        if spec.pattern.search(str(label)) is not None:
            return index
    return None


def accumulate_category_votes(
    evidences: list[VoteEvidence],
    evidence_categories: list[int],
    *,
    gaussian_count: int,
    category_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate sparse lifted mask votes into `(N, C)` vote and support tables."""

    if len(evidences) != len(evidence_categories):
        raise ValueError("evidences and evidence_categories must have the same length.")
    votes = np.zeros((gaussian_count, category_count), dtype=np.float32)
    support_counts = np.zeros((gaussian_count, category_count), dtype=np.int32)
    for evidence, category_index in zip(evidences, evidence_categories, strict=True):
        if category_index < 0 or category_index >= category_count:
            raise ValueError(f"Category index {category_index} is outside range.")
        if evidence.gaussian_indices.size == 0:
            continue
        np.add.at(votes[:, category_index], evidence.gaussian_indices, evidence.weights.astype(np.float32))
        np.add.at(
            support_counts[:, category_index],
            evidence.gaussian_indices,
            np.ones((evidence.gaussian_indices.size,), dtype=np.int32),
        )
    return votes, support_counts


def build_category_instance_ids(
    *,
    xyz: np.ndarray,
    category_votes: np.ndarray,
    support_counts: np.ndarray,
    category_names: list[str],
    background_support_counts: np.ndarray | None = None,
    min_vote_weight: float,
    min_support_count: int,
    min_foreground_ratio: float = 0.0,
    voxel_size: float,
    min_voxel_count: int,
    min_gaussians: int,
    max_instances_per_category: int = 0,
) -> tuple[np.ndarray, list[CategoryInstanceProposal]]:
    """Cluster high-vote Gaussians inside each category into instance candidates."""

    xyz = np.asarray(xyz, dtype=np.float32)
    category_votes = np.asarray(category_votes, dtype=np.float32)
    support_counts = np.asarray(support_counts, dtype=np.int32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {xyz.shape}")
    if category_votes.shape != support_counts.shape:
        raise ValueError("category_votes and support_counts must have the same shape.")
    if category_votes.shape[0] != xyz.shape[0]:
        raise ValueError("vote tables and xyz must have the same Gaussian count.")
    if category_votes.shape[1] != len(category_names):
        raise ValueError("category_names length must match vote table category count.")
    if background_support_counts is not None and background_support_counts.shape != (xyz.shape[0],):
        raise ValueError(
            f"background_support_counts must have shape ({xyz.shape[0]},), got {background_support_counts.shape}"
        )
    if min_foreground_ratio < 0.0 or min_foreground_ratio > 1.0:
        raise ValueError("min_foreground_ratio must be in [0, 1].")
    if max_instances_per_category < 0:
        raise ValueError("max_instances_per_category must be non-negative.")

    instance_ids = np.full((xyz.shape[0],), -1, dtype=np.int32)
    proposals: list[CategoryInstanceProposal] = []
    next_instance_id = 0

    best_category = np.argmax(category_votes, axis=1).astype(np.int32)
    best_vote = category_votes[np.arange(xyz.shape[0]), best_category]
    best_support = support_counts[np.arange(xyz.shape[0]), best_category]
    foreground_ratio = np.ones((xyz.shape[0],), dtype=np.float32)
    if background_support_counts is not None:
        denominator = best_support.astype(np.float32) + background_support_counts.astype(np.float32)
        valid_denominator = denominator > 0.0
        foreground_ratio[valid_denominator] = best_support[valid_denominator].astype(np.float32) / denominator[
            valid_denominator
        ]

    for category_index, category_name in enumerate(category_names):
        category_mask = (
            (best_category == category_index)
            & (best_vote >= float(min_vote_weight))
            & (best_support >= int(min_support_count))
            & (foreground_ratio >= float(min_foreground_ratio))
        )
        category_indices = np.flatnonzero(category_mask)
        if category_indices.size == 0:
            continue
        raw_components = build_voxel_connected_components(
            xyz[category_indices],
            voxel_size=voxel_size,
            min_voxel_count=min_voxel_count,
        )
        local_components = filter_and_remap_components(raw_components, min_gaussians=min_gaussians)
        if max_instances_per_category > 0:
            local_components = np.where(
                local_components < int(max_instances_per_category),
                local_components,
                -1,
            ).astype(np.int32)
        summaries = summarize_cluster_proposals(local_components, xyz[category_indices], voxel_size=voxel_size)
        local_to_global: dict[int, int] = {}
        for summary in summaries:
            local_id = int(summary.global_object_id)
            global_id = next_instance_id
            next_instance_id += 1
            local_to_global[local_id] = global_id
            member_indices = category_indices[local_components == local_id]
            proposals.append(
                CategoryInstanceProposal(
                    instance_id=global_id,
                    category=category_name,
                    rank_in_category=local_id,
                    gaussian_count=int(summary.gaussian_count),
                    voxel_count=int(summary.voxel_count),
                    vote_weight_sum=float(category_votes[member_indices, category_index].sum()),
                    support_count_mean=float(support_counts[member_indices, category_index].mean()),
                    centroid_xyz=summary.centroid_xyz,
                    bbox_min_xyz=summary.bbox_min_xyz,
                    bbox_max_xyz=summary.bbox_max_xyz,
                )
            )

        selected = local_components >= 0
        if np.any(selected):
            instance_ids[category_indices[selected]] = np.asarray(
                [local_to_global[int(local_id)] for local_id in local_components[selected]],
                dtype=np.int32,
            )
    return instance_ids, proposals


def save_category_instance_summary(
    *,
    output_path: Path,
    proposals: list[CategoryInstanceProposal],
    parameters: dict,
) -> None:
    """Write first-pass discovery summaries to JSON."""

    payload = {
        "instance_count": len(proposals),
        "parameters": parameters,
        "instances": [
            {
                "instance_id": proposal.instance_id,
                "category": proposal.category,
                "rank_in_category": proposal.rank_in_category,
                "gaussian_count": proposal.gaussian_count,
                "voxel_count": proposal.voxel_count,
                "vote_weight_sum": proposal.vote_weight_sum,
                "support_count_mean": proposal.support_count_mean,
                "centroid_xyz": proposal.centroid_xyz.astype(float).tolist(),
                "bbox_min_xyz": proposal.bbox_min_xyz.astype(float).tolist(),
                "bbox_max_xyz": proposal.bbox_max_xyz.astype(float).tolist(),
            }
            for proposal in proposals
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_category_instance_highlight_ply(
    *,
    output_path: Path,
    table: np.ndarray,
    header_properties: list[tuple[str, str]],
    instance_ids: np.ndarray,
    dim_opacity_scale: float,
) -> None:
    """Write a SuperSplat-compatible PLY with instance-colored candidates."""

    write_identity_instance_highlight_ply(
        output_path=output_path,
        table=table,
        header_properties=header_properties,
        instance_ids=instance_ids,
        dim_opacity_scale=dim_opacity_scale,
    )
