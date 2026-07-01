"""Interactive Gaussian-group transforms.

The editor needs two closely related edit modes:

1. World transform:
   Apply a rigid transform directly in the world frame.
2. Object-local transform:
   Rotate an object around its own pivot (typically its centroid), then place it
   back into the world with an additional translation.

We keep both explicit because mixing these frames silently is one of the easiest
ways to break object-level editing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_object_transform_about_pivot(
    means: torch.Tensor,
    rotations: torch.Tensor,
    object_ids: torch.Tensor,
    target_id: int,
    translation: torch.Tensor,
    rotation_matrix: torch.Tensor,
    pivot_xyz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an object-local SE(3) transform around a pivot point.

    Args:
        means: `(N, 3)` Gaussian centers in world coordinates.
        rotations: `(N, 4)` Gaussian quaternions in `(w, x, y, z)` order.
        object_ids: `(N,)` integer object ids aligned with `means`.
        target_id: Object id to edit.
        translation: `(3,)` world-space translation after the local rotation.
        rotation_matrix: `(3, 3)` rotation matrix in world coordinates.
        pivot_xyz: Optional `(3,)` pivot point in world coordinates. Defaults to
            the centroid of the selected object's Gaussian centers.
    """

    mask = object_ids == target_id
    if not mask.any():
        return means.clone(), rotations.clone()

    if pivot_xyz is None:
        pivot_xyz = means[mask].mean(dim=0)
    if pivot_xyz.shape != (3,):
        raise ValueError(f"Expected pivot_xyz shape (3,), got {pivot_xyz.shape}")

    pivot_xyz = pivot_xyz.to(device=means.device, dtype=means.dtype)
    translation = translation.to(device=means.device, dtype=means.dtype)

    centered_means = means.clone()
    centered_means[mask] = centered_means[mask] - pivot_xyz
    return apply_object_transform(
        centered_means,
        rotations,
        object_ids,
        target_id,
        translation=pivot_xyz + translation,
        rotation_matrix=rotation_matrix,
    )


def apply_object_transform(
    means: torch.Tensor,
    rotations: torch.Tensor,
    object_ids: torch.Tensor,
    target_id: int,
    translation: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a world-frame SE(3) transform to a specific object's Gaussians.

    Args:
        means: `(N, 3)` tensor of Gaussian centers.
        rotations: `(N, 4)` tensor of Gaussian rotations as quaternions
            `(w, x, y, z)`.
        object_ids: `(N,)` tensor of integer object labels.
        target_id: Integer object id to transform.
        translation: `(3,)` world-space displacement vector.
        rotation_matrix: `(3, 3)` world-space rotation matrix.

    Returns:
        A tuple `(new_means, new_rotations)` with non-selected Gaussians
        preserved exactly.
    """

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected means shape (N, 3), got {means.shape}")
    if rotations.ndim != 2 or rotations.shape[1] != 4:
        raise ValueError(f"Expected rotations shape (N, 4), got {rotations.shape}")
    if object_ids.ndim != 1:
        raise ValueError(f"Expected object_ids shape (N,), got {object_ids.shape}")
    if translation.shape != (3,):
        raise ValueError(f"Expected translation shape (3,), got {translation.shape}")
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation_matrix shape (3, 3), got {rotation_matrix.shape}")
    if means.shape[0] != rotations.shape[0] or means.shape[0] != object_ids.shape[0]:
        raise ValueError("N dimension mismatch across means, rotations, and object_ids")

    device = means.device
    translation = translation.to(device=device, dtype=means.dtype)
    rotation_matrix = rotation_matrix.to(device=device, dtype=means.dtype)

    new_means = means.clone()
    new_rotations = rotations.clone()

    mask = object_ids == target_id
    if not mask.any():
        # No-op keeps editing reversible when a stale object id is requested.
        return new_means, new_rotations

    target_means = means[mask]
    transformed_means = torch.matmul(target_means, rotation_matrix.transpose(0, 1)) + translation
    new_means[mask] = transformed_means

    m00, m01, m02 = rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2]
    m10, m11, m12 = rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2]
    m20, m21, m22 = rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        scale = torch.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif (m00 > m11) and (m00 > m22):
        scale = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    rotation_quat = torch.tensor([qw, qx, qy, qz], dtype=rotations.dtype, device=device)
    rotation_quat = F.normalize(rotation_quat.unsqueeze(0), p=2, dim=1)

    target_rotations = rotations[mask]
    w1, x1, y1, z1 = rotation_quat[0, 0], rotation_quat[0, 1], rotation_quat[0, 2], rotation_quat[0, 3]
    w2, x2, y2, z2 = (
        target_rotations[:, 0],
        target_rotations[:, 1],
        target_rotations[:, 2],
        target_rotations[:, 3],
    )

    new_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    new_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    new_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    new_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    updated_quats = torch.stack([new_w, new_x, new_y, new_z], dim=-1)
    updated_quats = F.normalize(updated_quats, p=2, dim=1)
    new_rotations[mask] = updated_quats

    if torch.isnan(new_means).any() or torch.isnan(new_rotations).any():
        raise RuntimeError("NaN introduced during Gaussian transformation.")

    return new_means, new_rotations
