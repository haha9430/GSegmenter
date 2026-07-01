"""Object proposal helpers built from multiview Gaussian vote evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gsegmenter.mapping.association import LocalInstanceEvidence
from gsegmenter.mapping.gaussian_io import rgb_to_sh_dc, write_gaussian_table


@dataclass(slots=True)
class ObjectProposal:
    """Summary for one discovered object proposal.

    The proposal id is a scene-global connected-component id produced before
    identity training. Gaussian indices are aligned with the source PLY row
    order, and all coordinates use the exported Gaussian world frame.
    """

    proposal_id: int
    label_family: str | None
    label: str | None
    gaussian_count: int
    support_frame_count: int
    local_instance_count: int
    total_vote_weight: float
    centroid_xyz: np.ndarray
    bbox_min_xyz: np.ndarray
    bbox_max_xyz: np.ndarray


def summarize_object_proposals(
    *,
    gaussian_object_ids: np.ndarray,
    gaussian_xyz: np.ndarray,
    local_instances: list[LocalInstanceEvidence],
    global_object_ids: np.ndarray,
) -> list[ObjectProposal]:
    """Build proposal summaries with majority label metadata."""

    proposals: list[ObjectProposal] = []
    positive_ids = sorted(int(value) for value in np.unique(gaussian_object_ids) if int(value) >= 0)
    for proposal_id in positive_ids:
        gaussian_mask = gaussian_object_ids == proposal_id
        proposal_xyz = gaussian_xyz[gaussian_mask]
        proposal_instances = [
            instance
            for instance in local_instances
            if int(global_object_ids[instance.local_index]) == proposal_id
        ]
        label_family = _most_common(instance.label_family for instance in proposal_instances)
        label = _most_common(instance.label for instance in proposal_instances)
        support_frames = {int(instance.frame_index) for instance in proposal_instances}
        total_vote_weight = float(sum(float(instance.weights.sum()) for instance in proposal_instances))
        proposals.append(
            ObjectProposal(
                proposal_id=proposal_id,
                label_family=label_family,
                label=label,
                gaussian_count=int(proposal_xyz.shape[0]),
                support_frame_count=len(support_frames),
                local_instance_count=len(proposal_instances),
                total_vote_weight=total_vote_weight,
                centroid_xyz=proposal_xyz.mean(axis=0),
                bbox_min_xyz=proposal_xyz.min(axis=0),
                bbox_max_xyz=proposal_xyz.max(axis=0),
            )
        )
    return proposals


def save_object_proposals(
    *,
    proposals: list[ObjectProposal],
    output_path: Path,
    parameters: dict | None = None,
) -> None:
    """Write proposal summaries to JSON for inspection and later approval."""

    payload = {
        "proposal_count": len(proposals),
        "parameters": parameters or {},
        "proposals": [
            {
                "proposal_id": proposal.proposal_id,
                "label_family": proposal.label_family,
                "label": proposal.label,
                "gaussian_count": proposal.gaussian_count,
                "support_frame_count": proposal.support_frame_count,
                "local_instance_count": proposal.local_instance_count,
                "total_vote_weight": proposal.total_vote_weight,
                "centroid_xyz": proposal.centroid_xyz.astype(float).tolist(),
                "bbox_min_xyz": proposal.bbox_min_xyz.astype(float).tolist(),
                "bbox_max_xyz": proposal.bbox_max_xyz.astype(float).tolist(),
            }
            for proposal in proposals
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_proposal_highlight_ply(
    *,
    output_path: Path,
    table: np.ndarray,
    header_properties: list[tuple[str, str]],
    proposal_ids: np.ndarray,
    keep_proposal_ids: set[int] | None = None,
    dim_opacity_scale: float = 0.35,
) -> None:
    """Color selected proposal Gaussians while preserving 3DGS PLY channels."""

    required_channels = ("f_dc_0", "f_dc_1", "f_dc_2")
    if not all(channel in table.dtype.names for channel in required_channels):
        raise ValueError("Input PLY does not contain Gaussian SH DC color channels.")
    if proposal_ids.shape[0] != table.shape[0]:
        raise ValueError(f"Proposal id count {proposal_ids.shape[0]} does not match Gaussian count {table.shape[0]}")
    if dim_opacity_scale <= 0.0:
        raise ValueError("dim_opacity_scale must be positive.")

    highlighted = table.copy()
    selected = proposal_ids >= 0
    if keep_proposal_ids is not None:
        selected &= np.isin(proposal_ids, np.asarray(sorted(keep_proposal_ids), dtype=np.int32))

    dc = np.stack(
        [highlighted["f_dc_0"], highlighted["f_dc_1"], highlighted["f_dc_2"]],
        axis=1,
    ).astype(np.float32)
    for proposal_id in sorted(int(value) for value in np.unique(proposal_ids[selected])):
        proposal_mask = proposal_ids == proposal_id
        dc[proposal_mask] = rgb_to_sh_dc(_palette_rgb(proposal_id))[None, :]
    highlighted["f_dc_0"] = dc[:, 0]
    highlighted["f_dc_1"] = dc[:, 1]
    highlighted["f_dc_2"] = dc[:, 2]

    if "opacity" in highlighted.dtype.names:
        background = ~selected
        highlighted["opacity"][background] = highlighted["opacity"][background] + np.float32(np.log(dim_opacity_scale))

    write_gaussian_table(output_path, highlighted, header_properties)


def select_top_proposals(
    proposals: list[ObjectProposal],
    *,
    limit: int,
    min_gaussians: int,
    min_support_frames: int,
) -> set[int]:
    """Return ids for the largest proposals passing basic stability filters."""

    kept = [
        proposal
        for proposal in proposals
        if proposal.gaussian_count >= min_gaussians and proposal.support_frame_count >= min_support_frames
    ]
    kept.sort(key=lambda proposal: (proposal.gaussian_count, proposal.support_frame_count), reverse=True)
    if limit > 0:
        kept = kept[:limit]
    return {proposal.proposal_id for proposal in kept}


def _most_common(values) -> str | None:
    counter = Counter(value for value in values if value is not None)
    if not counter:
        return None
    return str(counter.most_common(1)[0][0])


def _palette_rgb(proposal_id: int) -> np.ndarray:
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
    return palette[abs(int(proposal_id)) % len(palette)]
