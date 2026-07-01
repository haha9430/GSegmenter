"""Projection helpers for lifting 3D Gaussians into image space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsegmenter.data.nerfstudio_scene import CameraIntrinsics, FrameRecord

try:
    import cv2
except ImportError:  # pragma: no cover - exercised through fallback path.
    cv2 = None


@dataclass(slots=True)
class ProjectionResult:
    """Projected 2D locations and visibility metadata."""

    image_points: np.ndarray
    camera_points: np.ndarray
    depths: np.ndarray
    valid_mask: np.ndarray


def _world_to_camera_points(points_world: np.ndarray, frame: FrameRecord) -> np.ndarray:
    """Transform world points `(N, 3)` into the camera frame."""

    world_to_camera = frame.world_to_camera
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return points_world @ rotation.T + translation


def project_world_points(
    points_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    frame: FrameRecord,
) -> ProjectionResult:
    """Project world-space points into a NerfStudio frame.

    Args:
        points_world: `(N, 3)` array of world coordinates.
        intrinsics: Shared camera intrinsics.
        frame: Frame pose with a `camera_to_world` transform.

    Returns:
        ProjectionResult containing image coordinates and a validity mask.
    """

    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"Expected `(N, 3)` world points, got {points_world.shape}")

    camera_points = _world_to_camera_points(points_world, frame)
    depths = camera_points[:, 2]
    positive_depth = depths > 0.0

    if intrinsics.camera_model == "PINHOLE":
        image_points = np.empty((points_world.shape[0], 2), dtype=np.float64)
        image_points[:, 0] = intrinsics.fl_x * (camera_points[:, 0] / depths) + intrinsics.cx
        image_points[:, 1] = intrinsics.fl_y * (camera_points[:, 1] / depths) + intrinsics.cy
    elif intrinsics.camera_model == "OPENCV_FISHEYE":
        image_points = _project_fisheye(camera_points, intrinsics)
    elif intrinsics.camera_model == "OPENCV":
        image_points = _project_opencv(camera_points, intrinsics)
    else:
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for non-pinhole camera projection. "
                "Install `opencv-python` to project OPENCV or OPENCV_FISHEYE cameras."
            )
        world_to_camera = frame.world_to_camera
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = translation.reshape(3, 1)
        camera_matrix = intrinsics.camera_matrix
        distortion = intrinsics.distortion_array

        if intrinsics.camera_model == "OPENCV_FISHEYE":
            object_points = points_world.reshape(-1, 1, 3)
            image_points, _ = cv2.fisheye.projectPoints(
                object_points,
                rvec,
                tvec,
                camera_matrix,
                distortion.reshape(-1, 1),
            )
            image_points = image_points.reshape(-1, 2)
        else:
            image_points, _ = cv2.projectPoints(
                points_world,
                rvec,
                tvec,
                camera_matrix,
                distortion if distortion.size > 0 else None,
            )
            image_points = image_points.reshape(-1, 2)

    in_bounds = (
        (image_points[:, 0] >= 0.0)
        & (image_points[:, 0] < intrinsics.width)
        & (image_points[:, 1] >= 0.0)
        & (image_points[:, 1] < intrinsics.height)
    )
    finite = np.isfinite(image_points).all(axis=1) & np.isfinite(depths)
    valid_mask = positive_depth & in_bounds & finite

    return ProjectionResult(
        image_points=image_points.astype(np.float32),
        camera_points=camera_points.astype(np.float32),
        depths=depths.astype(np.float32),
        valid_mask=valid_mask,
    )


def _project_fisheye(camera_points: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project camera-frame points with the OpenCV fisheye distortion model."""

    x = camera_points[:, 0] / camera_points[:, 2]
    y = camera_points[:, 1] / camera_points[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)

    k1, k2, k3, k4 = (list(intrinsics.distortion_params) + [0.0, 0.0, 0.0, 0.0])[:4]
    theta2 = theta * theta
    theta_d = theta * (
        1.0
        + k1 * theta2
        + k2 * theta2 * theta2
        + k3 * theta2 * theta2 * theta2
        + k4 * theta2 * theta2 * theta2 * theta2
    )

    scale = np.ones_like(r)
    nonzero = r > 1e-8
    scale[nonzero] = theta_d[nonzero] / r[nonzero]

    xd = x * scale
    yd = y * scale
    return np.stack(
        [
            intrinsics.fl_x * xd + intrinsics.cx,
            intrinsics.fl_y * yd + intrinsics.cy,
        ],
        axis=1,
    )


def _project_opencv(camera_points: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project camera-frame points with the OpenCV radial+tangential model."""

    x = camera_points[:, 0] / camera_points[:, 2]
    y = camera_points[:, 1] / camera_points[:, 2]
    r2 = x * x + y * y

    params = list(intrinsics.distortion_params) + [0.0] * 8
    k1, k2, p1, p2, k3, k4, k5, k6 = params[:8]
    radial_num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    radial_den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2
    radial = radial_num / radial_den

    xy2 = 2.0 * x * y
    x_distorted = x * radial + p1 * xy2 + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + p2 * xy2
    return np.stack(
        [
            intrinsics.fl_x * x_distorted + intrinsics.cx,
            intrinsics.fl_y * y_distorted + intrinsics.cy,
        ],
        axis=1,
    )
