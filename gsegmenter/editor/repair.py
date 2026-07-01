"""Local scene-repair helpers for Gaussian object editing.

These utilities do not attempt full relighting. Instead, they perform bounded
appearance cleanup inside the source region of a moved/removed object so the old
location becomes less visually distracting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SourceCleanupSummary:
    """Debug summary for a bounded source-region cleanup pass."""

    inner_count: int
    shell_count: int
    color_blend: float
    opacity_scale: float


def cleanup_source_region_appearance(
    table: np.ndarray,
    object_ids: np.ndarray,
    target_object_id: int,
    source_bbox_min_xyz: np.ndarray,
    source_bbox_max_xyz: np.ndarray,
    *,
    shell_margin: float = 0.12,
    color_blend: float = 0.85,
    opacity_scale: float = 0.85,
    zero_high_order_sh: bool = True,
    mode: str = "blend",
) -> tuple[np.ndarray, SourceCleanupSummary]:
    """Blend residual appearance in the source region toward nearby background.

    Args:
        table: Structured Gaussian table containing `x/y/z`, `f_dc_*`, optional
            `opacity`, and optional `f_rest_*`.
        object_ids: `(N,)` integer ids aligned with the Gaussian rows.
        target_object_id: Edited object's id. Rows with this id are never
            directly rewritten by the cleanup pass.
        source_bbox_min_xyz: `(3,)` source-space minimum corner.
        source_bbox_max_xyz: `(3,)` source-space maximum corner.
        shell_margin: Outer margin around the source bbox used to sample nearby
            background appearance.
        color_blend: Interpolation factor toward the shell median color.
        opacity_scale: Multiplicative factor applied to opacity logits in the
            source bbox to soften residual artifacts.
        zero_high_order_sh: Whether to zero `f_rest_*` for cleaned Gaussians to
            suppress view-dependent residue.
        mode: Either `"blend"` for color+opacity cleanup or `"opacity_only"`
            for a more conservative fade-out that leaves color untouched.
    """

    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = required.difference(table.dtype.names)
    if missing:
        raise ValueError(f"Gaussian table is missing required columns: {sorted(missing)}")
    if object_ids.shape[0] != table.shape[0]:
        raise ValueError(
            f"Object id count {object_ids.shape[0]} does not match Gaussian count {table.shape[0]}"
        )

    source_bbox_min_xyz = np.asarray(source_bbox_min_xyz, dtype=np.float32)
    source_bbox_max_xyz = np.asarray(source_bbox_max_xyz, dtype=np.float32)
    if source_bbox_min_xyz.shape != (3,) or source_bbox_max_xyz.shape != (3,):
        raise ValueError("Source bbox min/max must have shape (3,).")
    if not (0.0 <= color_blend <= 1.0):
        raise ValueError("color_blend must stay within [0, 1].")
    if opacity_scale <= 0.0:
        raise ValueError("opacity_scale must be positive.")
    if mode not in {"blend", "opacity_only"}:
        raise ValueError(f"Unsupported cleanup mode: {mode}")

    xyz = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
    non_target_mask = object_ids != int(target_object_id)

    inner_mask = np.logical_and(
        np.all(xyz >= source_bbox_min_xyz[None, :], axis=1),
        np.all(xyz <= source_bbox_max_xyz[None, :], axis=1),
    )
    inner_mask &= non_target_mask

    outer_min = source_bbox_min_xyz - np.float32(shell_margin)
    outer_max = source_bbox_max_xyz + np.float32(shell_margin)
    outer_mask = np.logical_and(
        np.all(xyz >= outer_min[None, :], axis=1),
        np.all(xyz <= outer_max[None, :], axis=1),
    )
    shell_mask = np.logical_and(outer_mask, ~inner_mask)
    shell_mask &= non_target_mask

    repaired = table.copy()
    inner_count = int(np.count_nonzero(inner_mask))
    shell_count = int(np.count_nonzero(shell_mask))
    if inner_count == 0:
        return repaired, SourceCleanupSummary(
            inner_count=inner_count,
            shell_count=shell_count,
            color_blend=float(color_blend),
            opacity_scale=float(opacity_scale),
        )

    if mode == "blend":
        if shell_count == 0:
            return repaired, SourceCleanupSummary(
                inner_count=inner_count,
                shell_count=shell_count,
                color_blend=float(color_blend),
                opacity_scale=float(opacity_scale),
            )
        shell_dc = np.stack(
            [table["f_dc_0"][shell_mask], table["f_dc_1"][shell_mask], table["f_dc_2"][shell_mask]],
            axis=1,
        ).astype(np.float32)
        shell_dc_median = np.median(shell_dc, axis=0)

        inner_dc = np.stack(
            [repaired["f_dc_0"][inner_mask], repaired["f_dc_1"][inner_mask], repaired["f_dc_2"][inner_mask]],
            axis=1,
        ).astype(np.float32)
        blended_dc = (1.0 - color_blend) * inner_dc + color_blend * shell_dc_median[None, :]
        repaired["f_dc_0"][inner_mask] = blended_dc[:, 0]
        repaired["f_dc_1"][inner_mask] = blended_dc[:, 1]
        repaired["f_dc_2"][inner_mask] = blended_dc[:, 2]

    if "opacity" in repaired.dtype.names and opacity_scale != 1.0:
        repaired["opacity"][inner_mask] = repaired["opacity"][inner_mask] + np.float32(np.log(opacity_scale))

    if zero_high_order_sh and mode == "blend":
        for property_name in repaired.dtype.names:
            if property_name.startswith("f_rest_"):
                repaired[property_name][inner_mask] = np.float32(0.0)

    return repaired, SourceCleanupSummary(
        inner_count=inner_count,
        shell_count=shell_count,
        color_blend=float(color_blend),
        opacity_scale=float(opacity_scale),
    )
