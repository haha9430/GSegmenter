"""Factory helpers for identity-aware NerfStudio training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from gsegmenter.conf import IdentityTrainingConfig as AppIdentityTrainingConfig
from gsegmenter.conf import TrainingConfig as AppTrainingConfig
from gsegmenter.training.identity_datamanager import IdentityFullImageDatamanagerConfig
from gsegmenter.training.identity_dataset import build_identity_vocabulary, discover_mask_manifests
from gsegmenter.training.identity_splatfacto import HAS_NERFSTUDIO, IdentitySplatfactoModelConfig

if HAS_NERFSTUDIO:  # pragma: no branch
    from nerfstudio.configs.base_config import ViewerConfig
    from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.engine.optimizers import AdamOptimizerConfig
    from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
    from nerfstudio.engine.trainer import TrainerConfig
    from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig


def build_identity_optimizer_config(*, identity_only: bool = False) -> Dict[str, Dict]:
    """Mirror Splatfacto defaults while adding identity-specific parameter groups."""

    if not HAS_NERFSTUDIO:
        return {}

    identity_optimizers = {
        "identity_embeddings": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
        },
        "identity_field": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
        },
    }
    if identity_only:
        return identity_optimizers

    return {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        **identity_optimizers,
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    }


def infer_identity_num_classes(masks_root: Path, *, min_mask_score: float) -> int:
    """Infer the scene-global identity class count from mask manifests."""

    manifests = discover_mask_manifests(Path(masks_root))
    vocabulary = build_identity_vocabulary(manifests, min_score=min_mask_score)
    return max(vocabulary.num_classes, 1)


def build_identity_splatfacto_trainer_config(
    *,
    data_path: Path,
    masks_root: Path,
    training: AppTrainingConfig | None = None,
    identity: AppIdentityTrainingConfig | None = None,
    scene_format: str = "auto",
    downscale_factor: int = 1,
    downscale_rounding_mode: str = "floor",
    images_path: Path = Path("images"),
    colmap_path: Path = Path("colmap/sparse/0"),
    method_name: str = "identity-splatfacto",
    sh_degree: int = 3,
    identity_only: bool = False,
    load_checkpoint: Path | None = None,
) -> "TrainerConfig":
    """Build a trainer config that feeds identity labels into local Splatfacto."""

    if not HAS_NERFSTUDIO:
        raise ImportError("Identity trainer config requires nerfstudio to be installed.")

    training = training or AppTrainingConfig()
    identity = identity or AppIdentityTrainingConfig(enabled=True)
    num_classes = infer_identity_num_classes(Path(masks_root), min_mask_score=identity.min_mask_score)
    data_path = Path(data_path)
    if scene_format == "auto":
        resolved_scene_format = "nerfstudio" if (data_path / "transforms.json").exists() else "colmap"
    else:
        resolved_scene_format = scene_format
    if resolved_scene_format == "nerfstudio":
        dataparser = NerfstudioDataParserConfig(load_3D_points=True)
    elif resolved_scene_format == "colmap":
        dataparser = ColmapDataParserConfig(
            downscale_factor=int(downscale_factor),
            downscale_rounding_mode=downscale_rounding_mode,  # type: ignore[arg-type]
            images_path=Path(images_path),
            colmap_path=Path(colmap_path),
            eval_mode="all",
        )
    else:
        raise ValueError(f"Unsupported scene_format: {scene_format}")

    trainer_config = TrainerConfig(
        method_name=method_name,
        steps_per_eval_image=training.eval_interval,
        steps_per_eval_batch=0,
        steps_per_save=training.save_interval,
        steps_per_eval_all_images=max(training.eval_interval * 2, training.save_interval),
        max_num_iterations=training.num_iterations,
        mixed_precision=training.mixed_precision,
        pipeline=VanillaPipelineConfig(
            datamanager=IdentityFullImageDatamanagerConfig(
                data=data_path,
                dataparser=dataparser,
                cache_images_type="uint8",
                identity_masks_root=Path(masks_root),
                identity_min_mask_score=identity.min_mask_score,
            ),
            model=IdentitySplatfactoModelConfig(
                identity_enabled=identity.enabled,
                identity_embedding_dim=identity.embedding_dim,
                identity_num_classes=num_classes,
                identity_ignore_index=-1,
                cull_alpha_thresh=training.cull_alpha_thresh,
                cull_scale_thresh=training.cull_scale_thresh,
                reset_alpha_every=training.reset_alpha_every,
                densify_grad_thresh=training.densify_grad_thresh,
                use_scale_regularization=training.use_scale_regularization,
                max_gauss_ratio=training.max_gauss_ratio,
                sh_degree=int(sh_degree),
                identity_class_balance_power=identity.class_balance_power,
                identity_focal_gamma=identity.focal_gamma,
                identity_spatial_loss_weight=identity.spatial_loss_weight,
                identity_spatial_k_neighbors=identity.spatial_k_neighbors,
                identity_spatial_max_samples=identity.spatial_max_samples,
                identity_only=identity_only,
            ),
        ),
        optimizers=build_identity_optimizer_config(identity_only=identity_only),
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )
    trainer_config.load_checkpoint = Path(load_checkpoint) if load_checkpoint is not None else None
    return trainer_config
