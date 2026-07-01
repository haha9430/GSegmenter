"""Utilities for turning per-frame masks into training supervision tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gsegmenter.segmentation.mask_io import FrameMasksManifest, load_binary_mask, load_frame_masks_manifest


@dataclass(slots=True)
class IdentityLabelFrame:
    """Dense identity labels derived from a frame mask manifest.

    `label_map` uses `ignore_index` for background / unlabeled pixels and stores
    a contiguous training-class id per pixel elsewhere.
    """

    frame_index: int
    image_path: str
    label_map: np.ndarray
    class_ids: tuple[int, ...]
    ignore_index: int


def rasterize_identity_labels(
    manifest: FrameMasksManifest,
    masks_root: Path,
    *,
    min_score: float = 0.0,
    ignore_index: int = -1,
) -> IdentityLabelFrame:
    """Convert sparse instance masks into a dense integer label map.

    Masks are applied in descending score order so higher-confidence instances
    overwrite weaker ones on overlaps, matching the desired deterministic
    behavior for 2D supervision.
    """

    width, height = manifest.image_size
    label_map = np.full((height, width), ignore_index, dtype=np.int32)
    kept_instances = [instance for instance in manifest.instances if instance.score >= min_score]
    kept_instances.sort(key=lambda instance: float(instance.score))

    class_ids: list[int] = []
    class_lookup: dict[int, int] = {}
    for instance in kept_instances:
        if instance.instance_id not in class_lookup:
            class_lookup[instance.instance_id] = len(class_ids)
            class_ids.append(instance.instance_id)
        class_index = class_lookup[instance.instance_id]
        mask = load_binary_mask(masks_root / instance.mask_path)
        if mask.shape != label_map.shape:
            raise ValueError(
                f"Mask {instance.mask_path} has shape {mask.shape}, expected {label_map.shape}"
            )
        label_map[mask] = class_index

    return IdentityLabelFrame(
        frame_index=manifest.frame_index,
        image_path=manifest.image_path,
        label_map=label_map,
        class_ids=tuple(class_ids),
        ignore_index=ignore_index,
    )


def load_identity_label_frame(
    manifest_path: Path,
    *,
    min_score: float = 0.0,
    ignore_index: int = -1,
) -> IdentityLabelFrame:
    """Load and rasterize an on-disk frame mask manifest."""

    manifest_path = Path(manifest_path)
    manifest = load_frame_masks_manifest(manifest_path)
    return rasterize_identity_labels(
        manifest,
        manifest_path.parent,
        min_score=min_score,
        ignore_index=ignore_index,
    )
