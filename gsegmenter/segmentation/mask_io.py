"""Mask manifest helpers shared by extraction and lifting stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class MaskInstanceRecord:
    """Serializable metadata for a single 2D instance mask."""

    instance_id: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    area: int
    mask_path: str


@dataclass(slots=True)
class FrameMasksManifest:
    """Mask manifest for a single frame."""

    frame_index: int
    image_path: str
    image_size: tuple[int, int]
    instances: tuple[MaskInstanceRecord, ...]


def save_binary_mask(mask: np.ndarray, mask_path: Path) -> None:
    """Persist a boolean mask as an 8-bit PNG."""

    mask = np.asarray(mask, dtype=bool)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image.save(mask_path)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load an 8-bit PNG mask into a boolean array."""

    with Image.open(mask_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def save_frame_masks_manifest(manifest: FrameMasksManifest, output_path: Path) -> None:
    """Write the per-frame mask manifest as JSON."""

    payload = {
        "frame_index": manifest.frame_index,
        "image_path": manifest.image_path,
        "image_size": list(manifest.image_size),
        "instances": [asdict(instance) for instance in manifest.instances],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_frame_masks_manifest(path: Path) -> FrameMasksManifest:
    """Read a previously saved per-frame mask manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    instances = tuple(
        MaskInstanceRecord(
            instance_id=int(instance["instance_id"]),
            bbox_xyxy=tuple(int(value) for value in instance["bbox_xyxy"]),
            score=float(instance["score"]),
            area=int(instance["area"]),
            mask_path=str(instance["mask_path"]),
        )
        for instance in payload["instances"]
    )
    return FrameMasksManifest(
        frame_index=int(payload["frame_index"]),
        image_path=str(payload["image_path"]),
        image_size=tuple(int(value) for value in payload["image_size"]),
        instances=instances,
    )
