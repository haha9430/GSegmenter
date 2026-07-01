"""Execution helpers for identity-aware Splatfacto training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsegmenter.conf import AppConfig
from gsegmenter.training.identity_method import build_identity_splatfacto_trainer_config

try:  # pragma: no cover - exercised when nerfstudio is available.
    from nerfstudio.scripts.train import main as nerfstudio_train_main

    HAS_NERFSTUDIO_TRAIN = True
except ImportError:  # pragma: no cover - import-only fallback
    nerfstudio_train_main = None
    HAS_NERFSTUDIO_TRAIN = False


@dataclass(slots=True)
class IdentitySplatfactoTrainingSpec:
    """Resolved inputs for a single identity-aware Splatfacto training run."""

    data_path: Path
    masks_root: Path
    output_dir: Path
    scene_format: str = "auto"
    downscale_factor: int = 1
    downscale_rounding_mode: str = "floor"
    images_path: Path = Path("images")
    colmap_path: Path = Path("colmap/sparse/0")
    num_iterations: int = 30_000
    eval_interval: int = 500
    save_interval: int = 1_000
    mixed_precision: bool = False
    seed: int = 42
    quit_on_train_completion: bool = True
    cull_alpha_thresh: float = 0.1
    cull_scale_thresh: float = 0.5
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0008
    use_scale_regularization: bool = False
    max_gauss_ratio: float = 10.0
    sh_degree: int = 3
    identity_min_mask_score: float = 0.5
    identity_only: bool = False
    load_checkpoint: Path | None = None


def resolve_identity_training_spec(config: AppConfig, *, masks_root: Path | None = None) -> IdentitySplatfactoTrainingSpec:
    """Resolve a runnable identity-training spec from the project config."""

    resolved_masks_root = masks_root or (config.training.output_root / config.dataset.scene_name / "masks")
    return IdentitySplatfactoTrainingSpec(
        data_path=config.dataset.scene_root,
        masks_root=resolved_masks_root,
        output_dir=config.training.output_root / config.dataset.scene_name,
        num_iterations=config.training.num_iterations,
        eval_interval=config.training.eval_interval,
        save_interval=config.training.save_interval,
        mixed_precision=config.training.mixed_precision,
        seed=config.training.seed,
        quit_on_train_completion=True,
        cull_alpha_thresh=config.training.cull_alpha_thresh,
        cull_scale_thresh=config.training.cull_scale_thresh,
        reset_alpha_every=config.training.reset_alpha_every,
        densify_grad_thresh=config.training.densify_grad_thresh,
        use_scale_regularization=config.training.use_scale_regularization,
        max_gauss_ratio=config.training.max_gauss_ratio,
    )


def validate_identity_training_spec(spec: IdentitySplatfactoTrainingSpec) -> None:
    """Fail fast when the identity-aware training inputs are incomplete."""

    if not spec.data_path.exists():
        raise FileNotFoundError(f"NerfStudio dataset root does not exist: {spec.data_path}")
    if not spec.masks_root.exists():
        raise FileNotFoundError(f"Identity masks root does not exist: {spec.masks_root}")
    if spec.load_checkpoint is not None and not spec.load_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint does not exist: {spec.load_checkpoint}")


def _empty_adam_state(*, lr: float, eps: float, parameter_count: int) -> dict[str, Any]:
    """Build a fresh Adam state dict compatible with a single optimizer group."""

    return {
        "state": {},
        "param_groups": [
            {
                "lr": float(lr),
                "betas": (0.9, 0.999),
                "eps": float(eps),
                "weight_decay": 0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "params": list(range(int(parameter_count))),
            }
        ],
    }


def prepare_identity_only_checkpoint(source_checkpoint: Path, output_dir: Path) -> Path:
    """Copy a baseline pipeline checkpoint with fresh identity-only optimizer state.

    NerfStudio's checkpoint loader always restores optimizer state alongside the
    pipeline. Identity-only fine-tuning intentionally changes optimizer groups,
    so the baseline geometry/RGB tensors are kept while optimizer state is reset
    to the two trainable identity groups.
    """

    import torch

    source_checkpoint = Path(source_checkpoint)
    target_dir = Path(output_dir) / "_identity_init"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_dir / f"{source_checkpoint.stem}_identity_only.ckpt"
    loaded_state = torch.load(source_checkpoint, map_location="cpu")
    loaded_state["step"] = -1
    loaded_state["optimizers"] = {
        "identity_embeddings": _empty_adam_state(lr=1e-3, eps=1e-15, parameter_count=1),
        "identity_field": _empty_adam_state(lr=1e-3, eps=1e-15, parameter_count=2),
    }
    loaded_state.pop("schedulers", None)
    torch.save(loaded_state, target_checkpoint)
    return target_checkpoint


def build_identity_trainer_config(spec: IdentitySplatfactoTrainingSpec):
    """Build the local TrainerConfig for identity-aware training."""

    config = AppConfig()
    config.dataset.data_root = spec.data_path.parent
    config.dataset.scene_name = spec.data_path.name
    config.training.output_root = spec.output_dir.parent
    config.training.num_iterations = spec.num_iterations
    config.training.eval_interval = spec.eval_interval
    config.training.save_interval = spec.save_interval
    config.training.mixed_precision = spec.mixed_precision
    config.training.seed = spec.seed
    config.training.cull_alpha_thresh = spec.cull_alpha_thresh
    config.training.cull_scale_thresh = spec.cull_scale_thresh
    config.training.reset_alpha_every = spec.reset_alpha_every
    config.training.densify_grad_thresh = spec.densify_grad_thresh
    config.training.use_scale_regularization = spec.use_scale_regularization
    config.training.max_gauss_ratio = spec.max_gauss_ratio
    config.identity_training.enabled = True
    config.identity_training.min_mask_score = spec.identity_min_mask_score

    load_checkpoint = spec.load_checkpoint
    if spec.identity_only and spec.load_checkpoint is not None:
        load_checkpoint = prepare_identity_only_checkpoint(spec.load_checkpoint, spec.output_dir)

    trainer_config = build_identity_splatfacto_trainer_config(
        data_path=spec.data_path,
        masks_root=spec.masks_root,
        training=config.training,
        identity=config.identity_training,
        scene_format=spec.scene_format,
        downscale_factor=spec.downscale_factor,
        downscale_rounding_mode=spec.downscale_rounding_mode,
        images_path=spec.images_path,
        colmap_path=spec.colmap_path,
        sh_degree=spec.sh_degree,
        identity_only=spec.identity_only,
        load_checkpoint=load_checkpoint,
    )
    trainer_config.output_dir = spec.output_dir
    trainer_config.machine.seed = spec.seed
    trainer_config.viewer.quit_on_train_completion = spec.quit_on_train_completion
    if spec.identity_only and spec.load_checkpoint is not None:
        trainer_config.load_scheduler = False
    return trainer_config


def run_identity_splatfacto_training(spec: IdentitySplatfactoTrainingSpec, *, dry_run: bool = False):
    """Run or preview the local identity-aware training path."""

    validate_identity_training_spec(spec)
    trainer_config = build_identity_trainer_config(spec)
    if dry_run:
        return trainer_config
    if nerfstudio_train_main is None:
        raise ImportError("nerfstudio.scripts.train is not available in the current Python environment.")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_train_main(trainer_config)
    return trainer_config
