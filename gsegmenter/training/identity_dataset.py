"""Scene-level identity supervision datasets for Gaussian Grouping style training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gsegmenter.segmentation.mask_io import load_frame_masks_manifest
from gsegmenter.training.identity_supervision import IdentityLabelFrame, rasterize_identity_labels


@dataclass(slots=True)
class IdentityClassVocabulary:
    """Scene-level mapping from raw tracked object ids to contiguous class ids."""

    raw_object_ids: tuple[int, ...]
    ignore_index: int = -1
    _to_class_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._to_class_index = {int(object_id): index for index, object_id in enumerate(self.raw_object_ids)}

    @property
    def num_classes(self) -> int:
        return len(self.raw_object_ids)

    def class_index(self, raw_object_id: int) -> int:
        return self._to_class_index[int(raw_object_id)]


@dataclass(slots=True)
class SceneIdentityLabelFrame:
    """Identity labels remapped into a scene-global class vocabulary."""

    frame_index: int
    image_path: str
    label_map: np.ndarray
    ignore_index: int


def discover_mask_manifests(masks_root: Path) -> tuple[Path, ...]:
    """Return all per-frame instance manifests under a mask root."""

    masks_root = Path(masks_root)
    manifests = tuple(sorted(masks_root.glob("frame_*/instances.json")))
    if not manifests:
        raise FileNotFoundError(f"No frame manifests were found under {masks_root}")
    return manifests


def build_identity_vocabulary(manifest_paths: tuple[Path, ...], *, min_score: float = 0.0) -> IdentityClassVocabulary:
    """Build a dataset-wide class vocabulary from tracked instance ids."""

    raw_ids: set[int] = set()
    for manifest_path in manifest_paths:
        manifest = load_frame_masks_manifest(manifest_path)
        for instance in manifest.instances:
            if instance.score >= min_score:
                raw_ids.add(int(instance.instance_id))
    return IdentityClassVocabulary(raw_object_ids=tuple(sorted(raw_ids)))


def remap_frame_to_scene_classes(
    frame_labels: IdentityLabelFrame,
    vocabulary: IdentityClassVocabulary,
) -> SceneIdentityLabelFrame:
    """Convert frame-local class indices into scene-global class indices."""

    remapped = np.full_like(frame_labels.label_map, vocabulary.ignore_index, dtype=np.int32)
    for local_index, raw_object_id in enumerate(frame_labels.class_ids):
        class_index = vocabulary.class_index(raw_object_id)
        remapped[frame_labels.label_map == local_index] = class_index
    return SceneIdentityLabelFrame(
        frame_index=frame_labels.frame_index,
        image_path=frame_labels.image_path,
        label_map=remapped,
        ignore_index=vocabulary.ignore_index,
    )


def load_scene_identity_frames(
    masks_root: Path,
    *,
    min_score: float = 0.0,
    ignore_index: int = -1,
) -> tuple[IdentityClassVocabulary, tuple[SceneIdentityLabelFrame, ...]]:
    """Load all frame labels under a mask root with a scene-global label vocabulary."""

    manifest_paths = discover_mask_manifests(masks_root)
    vocabulary = build_identity_vocabulary(manifest_paths, min_score=min_score)
    vocabulary.ignore_index = ignore_index
    frames: list[SceneIdentityLabelFrame] = []
    for manifest_path in manifest_paths:
        manifest = load_frame_masks_manifest(manifest_path)
        frame_labels = rasterize_identity_labels(
            manifest,
            manifest_path.parent,
            min_score=min_score,
            ignore_index=ignore_index,
        )
        frames.append(remap_frame_to_scene_classes(frame_labels, vocabulary))
    return vocabulary, tuple(frames)
