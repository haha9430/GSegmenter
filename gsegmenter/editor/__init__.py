"""Interactive object editing and scene-repair operations."""

from .occupancy import OccupancyPlacementSummary, evaluate_interiorgs_object_placement
from .repair import SourceCleanupSummary, cleanup_source_region_appearance
from .transform import apply_object_transform, apply_object_transform_about_pivot

__all__ = [
    "OccupancyPlacementSummary",
    "SourceCleanupSummary",
    "apply_object_transform",
    "apply_object_transform_about_pivot",
    "cleanup_source_region_appearance",
    "evaluate_interiorgs_object_placement",
]
