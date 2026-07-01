"""Gaussian training utilities and NerfStudio adapters."""

from gsegmenter.training.identity_bridge import (
    IdentityTrainingBatch,
    build_identity_training_batch,
    extract_rendered_identity_embeddings,
    gather_scene_identity_targets,
)
from gsegmenter.training.identity_dataset import (
    IdentityClassVocabulary,
    SceneIdentityLabelFrame,
    build_identity_vocabulary,
    discover_mask_manifests,
    load_scene_identity_frames,
    remap_frame_to_scene_classes,
)
from gsegmenter.training.identity_datamanager import (
    IdentityFullImageDatamanager,
    IdentityFullImageDatamanagerConfig,
    build_identity_frame_lookup,
    normalize_identity_image_path,
    prepare_identity_label_map,
    resolve_identity_frame,
)
from gsegmenter.training.identity_loss import IdentityLossBreakdown, compute_identity_training_loss
from gsegmenter.training.identity_method import (
    build_identity_optimizer_config,
    build_identity_splatfacto_trainer_config,
    infer_identity_num_classes,
)
from gsegmenter.training.identity_eval import load_identity_eval_setup
from gsegmenter.training.identity_export import (
    build_identity_export_tensors,
    export_identity_sidecar,
    filter_finite_and_visible_gaussians,
)
from gsegmenter.training.identity_runner import (
    IdentitySplatfactoTrainingSpec,
    build_identity_trainer_config,
    resolve_identity_training_spec,
    run_identity_splatfacto_training,
    validate_identity_training_spec,
)
from gsegmenter.training.identity_step import IdentityStepResult, run_identity_optimization_step
from gsegmenter.training.identity_splatfacto import (
    IdentitySplatfactoModel,
    IdentitySplatfactoModelConfig,
    prepare_identity_label_tensor,
    render_identity_channels,
)
from gsegmenter.training.identity_supervision import (
    IdentityLabelFrame,
    load_identity_label_frame,
    rasterize_identity_labels,
)
from gsegmenter.training.object_field import GaussianIdentityField, IdentityFieldOutput
from gsegmenter.training.regularization import identity_spatial_consistency_loss
from gsegmenter.training.splatfacto import (
    SplatfactoTrainingSpec,
    build_ns_train_command,
    resolve_splatfacto_spec,
    run_splatfacto_training,
    validate_training_spec,
)

__all__ = [
    "GaussianIdentityField",
    "IdentityFullImageDatamanager",
    "IdentityFullImageDatamanagerConfig",
    "IdentityFieldOutput",
    "IdentityClassVocabulary",
    "IdentityLabelFrame",
    "IdentityLossBreakdown",
    "IdentitySplatfactoModel",
    "IdentitySplatfactoModelConfig",
    "IdentitySplatfactoTrainingSpec",
    "IdentityStepResult",
    "IdentityTrainingBatch",
    "SceneIdentityLabelFrame",
    "SplatfactoTrainingSpec",
    "build_identity_frame_lookup",
    "build_identity_optimizer_config",
    "build_identity_splatfacto_trainer_config",
    "build_identity_export_tensors",
    "export_identity_sidecar",
    "filter_finite_and_visible_gaussians",
    "load_identity_eval_setup",
    "build_identity_vocabulary",
    "build_identity_training_batch",
    "build_ns_train_command",
    "compute_identity_training_loss",
    "discover_mask_manifests",
    "extract_rendered_identity_embeddings",
    "gather_scene_identity_targets",
    "identity_spatial_consistency_loss",
    "infer_identity_num_classes",
    "build_identity_trainer_config",
    "load_identity_label_frame",
    "load_scene_identity_frames",
    "normalize_identity_image_path",
    "prepare_identity_label_map",
    "prepare_identity_label_tensor",
    "rasterize_identity_labels",
    "render_identity_channels",
    "remap_frame_to_scene_classes",
    "resolve_identity_frame",
    "resolve_identity_training_spec",
    "resolve_splatfacto_spec",
    "run_identity_splatfacto_training",
    "run_identity_optimization_step",
    "run_splatfacto_training",
    "validate_identity_training_spec",
    "validate_training_spec",
]
