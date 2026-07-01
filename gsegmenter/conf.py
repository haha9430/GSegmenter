"""Typed configuration objects for the NerfStudio-first project layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DatasetConfig:
    """COLMAP-aligned scene inputs.

    Paths are kept relative by default so they can be resolved from the repo
    root or overridden by a launcher without baking in machine-specific paths.
    """

    data_root: Path = Path("data")
    scene_name: str = "scene01"
    images_subdir: str = "images"
    colmap_subdir: str = "colmap"
    use_sparse_colmap: bool = True

    @property
    def scene_root(self) -> Path:
        return self.data_root / self.scene_name

    @property
    def images_dir(self) -> Path:
        return self.scene_root / self.images_subdir

    @property
    def colmap_dir(self) -> Path:
        return self.scene_root / self.colmap_subdir


@dataclass(slots=True)
class TrainingConfig:
    """Training parameters for NerfStudio Splatfacto runs."""

    method: str = "splatfacto"
    output_root: Path = Path("outputs")
    num_iterations: int = 30_000
    seed: int = 42
    eval_interval: int = 500
    save_interval: int = 1_000
    mixed_precision: bool = False
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0008
    use_scale_regularization: bool = False
    max_gauss_ratio: float = 10.0


@dataclass(slots=True)
class IdentityTrainingConfig:
    """Identity-aware training settings inspired by Gaussian Grouping."""

    enabled: bool = False
    embedding_dim: int = 16
    max_objects_per_frame: int = 64
    min_mask_score: float = 0.5
    class_balance_power: float = 0.5
    focal_gamma: float = 0.0
    spatial_k_neighbors: int = 8
    spatial_loss_weight: float = 0.1
    spatial_max_samples: int = 4096
    flatten_untracked_background: bool = False


@dataclass(slots=True)
class MappingConfig:
    """Multi-view mask lifting and voting settings."""

    vote_threshold: float = 0.6
    min_visible_views: int = 2
    max_inlier_distance_px: float = 2.0
    use_deterministic_voting: bool = True


@dataclass(slots=True)
class EditorConfig:
    """Interactive editing and scene repair settings."""

    enable_infilling: bool = True
    allow_non_rigid: bool = False
    debug_visualization: bool = False


@dataclass(slots=True)
class AppConfig:
    """Top-level configuration used to wire the first project scaffold."""

    project_name: str = "gsegmenter"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    identity_training: IdentityTrainingConfig = field(default_factory=IdentityTrainingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
