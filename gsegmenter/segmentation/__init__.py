"""Mask extraction and 2D segmentation adapters."""

from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    load_binary_mask,
    load_frame_masks_manifest,
    save_binary_mask,
    save_frame_masks_manifest,
)
from gsegmenter.segmentation.sam2_extractor import (
    AutomaticMaskPrediction,
    Sam2AutomaticMaskExtractor,
)
from gsegmenter.segmentation.sam2_runner import (
    Sam2ExtractionSpec,
    build_sam2_extract_command,
    run_sam2_extraction,
    validate_sam2_spec,
)

__all__ = [
    "AutomaticMaskPrediction",
    "FrameMasksManifest",
    "MaskInstanceRecord",
    "Sam2AutomaticMaskExtractor",
    "Sam2ExtractionSpec",
    "build_sam2_extract_command",
    "load_binary_mask",
    "load_frame_masks_manifest",
    "run_sam2_extraction",
    "save_binary_mask",
    "save_frame_masks_manifest",
    "validate_sam2_spec",
]
