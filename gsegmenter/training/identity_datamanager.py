"""Identity-label aware datamanager adapters for Splatfacto training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from copy import deepcopy
from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F

from gsegmenter.training.identity_dataset import SceneIdentityLabelFrame, load_scene_identity_frames

try:  # pragma: no cover - exercised when nerfstudio is available.
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.data.datamanagers.full_images_datamanager import (
        FullImageDatamanager,
        FullImageDatamanagerConfig,
    )

    HAS_NERFSTUDIO = True
except ImportError:  # pragma: no cover - base interpreter tests use fallbacks.
    Cameras = object  # type: ignore[assignment]
    FullImageDatamanager = object  # type: ignore[assignment]
    FullImageDatamanagerConfig = object  # type: ignore[assignment]
    HAS_NERFSTUDIO = False


def normalize_identity_image_path(image_path: str | Path) -> str:
    """Normalize image paths so NerfStudio filenames and mask manifests can match."""

    return Path(image_path).as_posix().lstrip("./")


def build_identity_frame_lookup(
    frames: tuple[SceneIdentityLabelFrame, ...],
) -> tuple[Dict[str, SceneIdentityLabelFrame], Dict[int, SceneIdentityLabelFrame]]:
    """Build lookup tables keyed by normalized image path and frame index."""

    by_path = {normalize_identity_image_path(frame.image_path): frame for frame in frames}
    by_index = {int(frame.frame_index): frame for frame in frames}
    return by_path, by_index


def resolve_identity_frame(
    image_filename: str | Path,
    dataset_index: int,
    *,
    by_path: Dict[str, SceneIdentityLabelFrame],
    by_index: Dict[int, SceneIdentityLabelFrame],
) -> Optional[SceneIdentityLabelFrame]:
    """Resolve the label frame corresponding to a NerfStudio dataset sample."""

    normalized = normalize_identity_image_path(image_filename)
    if normalized in by_path:
        return by_path[normalized]
    basename = Path(normalized).name
    for key, frame in by_path.items():
        if Path(key).name == basename:
            return frame
    return by_index.get(int(dataset_index))


def prepare_identity_label_map(
    label_map: np.ndarray | torch.Tensor,
    image_hw: tuple[int, int],
    *,
    ignore_index: int = -1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Prepare a `(H, W)` integer label tensor aligned with a cached image shape.

    The current path assumes masks were generated from the same image geometry as
    training. If the cached image size differs, we apply nearest-neighbor resize
    to avoid silent broadcasting bugs while preserving integer ids.
    """

    labels = torch.as_tensor(label_map, dtype=torch.int64)
    if labels.ndim != 2:
        raise ValueError(f"Expected `(H, W)` identity labels, got shape {tuple(labels.shape)}")

    target_h, target_w = int(image_hw[0]), int(image_hw[1])
    if tuple(labels.shape) != (target_h, target_w):
        labels = F.interpolate(
            labels.to(torch.float32).unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(torch.int64)
    labels[labels < ignore_index] = ignore_index
    if device is not None:
        labels = labels.to(device)
    return labels


if HAS_NERFSTUDIO:  # pragma: no branch

    @dataclass
    class IdentityFullImageDatamanagerConfig(FullImageDatamanagerConfig):
        """Full-image datamanager extended with scene-global identity labels."""

        _target: Type = field(default_factory=lambda: IdentityFullImageDatamanager)
        identity_masks_root: Optional[Path] = None
        identity_min_mask_score: float = 0.5
        identity_ignore_index: int = -1


    class IdentityFullImageDatamanager(FullImageDatamanager):
        """Attach Gaussian Grouping style identity labels to image batches."""

        config: IdentityFullImageDatamanagerConfig

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
            self.identity_vocabulary = None
            self.identity_frames: tuple[SceneIdentityLabelFrame, ...] = ()
            self.identity_frames_by_path: Dict[str, SceneIdentityLabelFrame] = {}
            self.identity_frames_by_index: Dict[int, SceneIdentityLabelFrame] = {}
            if self.config.identity_masks_root is not None:
                vocabulary, frames = load_scene_identity_frames(
                    Path(self.config.identity_masks_root),
                    min_score=self.config.identity_min_mask_score,
                    ignore_index=self.config.identity_ignore_index,
                )
                self.identity_vocabulary = vocabulary
                self.identity_frames = frames
                self.identity_frames_by_path, self.identity_frames_by_index = build_identity_frame_lookup(frames)

        def _dataparser_image_path(self, split: str, image_idx: int) -> Path:
            if split == "train":
                return Path(self.train_dataparser_outputs.image_filenames[image_idx])
            return Path(self.eval_dataparser_outputs.image_filenames[image_idx])

        def _attach_identity_labels(self, data: Dict, *, split: str, image_idx: int) -> Dict:
            if not self.identity_frames:
                return data

            frame = resolve_identity_frame(
                self._dataparser_image_path(split, image_idx),
                image_idx,
                by_path=self.identity_frames_by_path,
                by_index=self.identity_frames_by_index,
            )
            if frame is None:
                return data

            labels = prepare_identity_label_map(
                frame.label_map,
                data["image"].shape[:2],
                ignore_index=frame.ignore_index,
                device=self.device,
            )
            data["identity_labels"] = labels
            data["identity_frame_index"] = torch.tensor(frame.frame_index, dtype=torch.int64, device=self.device)
            return data

        def next_train(self, step: int) -> Tuple[Cameras, Dict]:
            camera, data = super().next_train(step)
            image_idx = int(camera.metadata["cam_idx"]) if camera.metadata is not None else -1
            data = self._attach_identity_labels(data, split="train", image_idx=image_idx)
            return camera, data

        def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
            camera, data = self.next_eval_image(step)
            return camera, data

        def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
            """Return eval batches with explicit camera indices in metadata."""

            image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
            if len(self.eval_unseen_cameras) == 0:
                self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]

            data = self.cached_eval[image_idx].copy()
            data["image"] = data["image"].to(self.device)
            assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
            camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = image_idx
            image_idx = int(camera.metadata["cam_idx"]) if camera.metadata is not None else -1
            data = self._attach_identity_labels(data, split="eval", image_idx=image_idx)
            return camera, data

        @property
        def fixed_indices_eval_dataloader(self):
            data = [d.copy() for d in self.cached_eval]
            cameras = []
            _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
            patched = []
            for image_idx in range(len(self.eval_dataset)):
                data[image_idx]["image"] = data[image_idx]["image"].to(self.device)
                camera = _cameras[image_idx : image_idx + 1]
                if camera.metadata is None:
                    camera.metadata = {}
                camera.metadata["cam_idx"] = image_idx
                cameras.append(camera)
            for image_idx, camera in enumerate(cameras):
                sample = data[image_idx]
                patched.append((camera, self._attach_identity_labels(sample, split="eval", image_idx=image_idx)))
            return patched


else:

    @dataclass
    class IdentityFullImageDatamanagerConfig:  # pragma: no cover - import-only fallback
        """Fallback config used when nerfstudio is unavailable."""

        identity_masks_root: Optional[Path] = None
        identity_min_mask_score: float = 0.5
        identity_ignore_index: int = -1


    class IdentityFullImageDatamanager:  # pragma: no cover - import-only fallback
        def __init__(self, *args, **kwargs):
            raise ImportError("IdentityFullImageDatamanager requires nerfstudio to be installed.")
