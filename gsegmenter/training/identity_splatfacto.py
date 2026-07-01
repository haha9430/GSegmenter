"""Identity-aware Splatfacto integration scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from gsegmenter.training.identity_loss import compute_identity_training_loss
from gsegmenter.training.object_field import GaussianIdentityField

try:  # pragma: no cover - exercised only when nerfstudio is available.
    from gsplat.rendering import rasterization
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.models.splatfacto import (
        SplatfactoModel,
        SplatfactoModelConfig,
        get_viewmat,
    )

    HAS_NERFSTUDIO = True
except ImportError:  # pragma: no cover - tests run without nerfstudio in the base interpreter.
    rasterization = None
    SplatfactoModel = object  # type: ignore[assignment]
    SplatfactoModelConfig = object  # type: ignore[assignment]
    Cameras = object  # type: ignore[assignment]
    get_viewmat = None
    HAS_NERFSTUDIO = False


def prepare_identity_label_tensor(
    label_map: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Resize integer identity labels to a rendered resolution using nearest neighbor."""

    if label_map.ndim == 2:
        label_map = label_map.unsqueeze(0)
    if label_map.ndim == 4 and label_map.shape[-1] == 1:
        label_map = label_map.squeeze(-1)
    if label_map.ndim != 3:
        raise ValueError(
            f"Expected identity labels with shape `(B, H, W)` or `(H, W)`, got {tuple(label_map.shape)}"
        )

    labels = label_map.to(torch.float32)
    labels = F.interpolate(labels[:, None, ...], size=target_hw, mode="nearest").squeeze(1)
    labels = labels.to(torch.int64)
    labels[labels < ignore_index] = ignore_index
    return labels


def render_identity_channels(
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    identity_embeddings: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    rasterize_mode: str,
    packed: bool = False,
) -> torch.Tensor:
    """Render per-pixel identity embeddings using gsplat's N-D feature support."""

    if rasterization is None:
        raise ImportError("gsplat/nerfstudio is not available in the current Python environment.")

    backgrounds = torch.zeros(
        (viewmats.shape[0], identity_embeddings.shape[-1]),
        dtype=identity_embeddings.dtype,
        device=identity_embeddings.device,
    )
    render_object, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=identity_embeddings,
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        packed=packed,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=rasterize_mode,
        backgrounds=backgrounds,
    )
    return render_object


if HAS_NERFSTUDIO:  # pragma: no branch

    @dataclass
    class IdentitySplatfactoModelConfig(SplatfactoModelConfig):
        """Splatfacto config extended with Gaussian Grouping style identity learning."""

        _target: Type = field(default_factory=lambda: IdentitySplatfactoModel)
        identity_enabled: bool = False
        identity_embedding_dim: int = 16
        identity_num_classes: int = 64
        identity_ignore_index: int = -1
        identity_loss_weight: float = 1.0
        identity_class_balance_power: float = 0.5
        identity_focal_gamma: float = 0.0
        identity_spatial_loss_weight: float = 0.1
        identity_spatial_k_neighbors: int = 8
        identity_spatial_max_samples: int = 4096
        identity_render_key: str = "render_object"
        identity_only: bool = False


    class IdentitySplatfactoModel(SplatfactoModel):
        """Local Splatfacto subclass that emits rendered identity embeddings."""

        config: IdentitySplatfactoModelConfig

        def populate_modules(self):
            super().populate_modules()
            if not self.config.identity_enabled:
                return
            identity_embeddings = torch.nn.Parameter(
                torch.zeros((self.num_points, self.config.identity_embedding_dim), dtype=torch.float32)
            )
            torch.nn.init.normal_(identity_embeddings, mean=0.0, std=0.01)
            self.gauss_params["identity_embeddings"] = identity_embeddings
            self.identity_field = GaussianIdentityField(
                num_gaussians=self.num_points,
                embedding_dim=self.config.identity_embedding_dim,
                num_classes=self.config.identity_num_classes,
            )
            self.identity_field.gaussian_embeddings = self.gauss_params["identity_embeddings"]

        @property
        def identity_embeddings(self) -> torch.Tensor:
            return self.gauss_params["identity_embeddings"]

        def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
            if self.config.identity_enabled and self.config.identity_only:
                return {"identity_embeddings": [self.gauss_params["identity_embeddings"]]}
            groups = super().get_gaussian_param_groups()
            if self.config.identity_enabled:
                groups["identity_embeddings"] = [self.gauss_params["identity_embeddings"]]
            return groups

        def get_param_groups(self) -> Dict[str, List[Parameter]]:
            if self.config.identity_enabled and self.config.identity_only:
                return {
                    "identity_embeddings": [self.gauss_params["identity_embeddings"]],
                    "identity_field": list(self.identity_field.classifier.parameters()),
                }
            groups = super().get_param_groups()
            if self.config.identity_enabled:
                groups["identity_field"] = list(self.identity_field.classifier.parameters())
            return groups

        def get_training_callbacks(self, training_callback_attributes):
            callbacks = super().get_training_callbacks(training_callback_attributes)
            if not (self.config.identity_enabled and self.config.identity_only):
                return callbacks
            return [
                callback
                for callback in callbacks
                if getattr(callback.func, "__name__", "") != "step_post_backward"
            ]

        def load_state_dict(self, dict, **kwargs):  # type: ignore
            if self.config.identity_enabled and "identity_field.gaussian_embeddings" in dict:
                dict.pop("identity_field.gaussian_embeddings")
            if self.config.identity_enabled and "gauss_params.identity_embeddings" not in dict:
                # Baseline Splatfacto checkpoints do not contain identity
                # attributes. Keep the randomly initialized per-Gaussian
                # embeddings instead of inserting a constant tensor; a constant
                # initialization makes identity-only training collapse to class
                # priors because all Gaussians start indistinguishable.
                kwargs["strict"] = False
            super().load_state_dict(dict, **kwargs)

        def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
            outputs = super().get_outputs(camera)
            if not self.config.identity_enabled:
                return outputs
            if not isinstance(camera, Cameras):
                return outputs

            if self.training:
                optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
            else:
                optimized_camera_to_world = camera.camera_to_worlds

            if self.crop_box is not None and not self.training:
                crop_ids = self.crop_box.within(self.means).squeeze()
                if crop_ids.sum() == 0:
                    outputs[self.config.identity_render_key] = torch.zeros(
                        (*outputs["rgb"].shape[:2], self.config.identity_embedding_dim),
                        dtype=outputs["rgb"].dtype,
                        device=outputs["rgb"].device,
                    )
                    return outputs
            else:
                crop_ids = None

            if crop_ids is not None:
                means_crop = self.means[crop_ids]
                scales_crop = self.scales[crop_ids]
                quats_crop = self.quats[crop_ids]
                opacities_crop = self.opacities[crop_ids]
                identity_embeddings_crop = self.identity_embeddings[crop_ids]
            else:
                means_crop = self.means
                scales_crop = self.scales
                quats_crop = self.quats
                opacities_crop = self.opacities
                identity_embeddings_crop = self.identity_embeddings

            camera_scale_fac = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_scale_fac)
            viewmat = get_viewmat(optimized_camera_to_world)
            intrinsics = camera.get_intrinsics_matrices().to(self.device)
            width, height = int(camera.width.item()), int(camera.height.item())
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

            render_object = render_identity_channels(
                means=means_crop,
                quats=quats_crop,
                scales=scales_crop,
                opacities=opacities_crop,
                identity_embeddings=identity_embeddings_crop,
                viewmats=viewmat,
                intrinsics=intrinsics,
                width=width,
                height=height,
                rasterize_mode=self.config.rasterize_mode,
                packed=False,
            )
            outputs[self.config.identity_render_key] = render_object.squeeze(0)
            return outputs

        def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
            loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
            if not self.config.identity_enabled:
                return loss_dict
            if self.config.identity_render_key not in outputs or "identity_labels" not in batch:
                return loss_dict

            pixel_embeddings = outputs[self.config.identity_render_key].permute(2, 0, 1).unsqueeze(0)
            target_labels = prepare_identity_label_tensor(
                batch["identity_labels"].to(pixel_embeddings.device),
                pixel_embeddings.shape[2:],
                ignore_index=self.config.identity_ignore_index,
            )
            self.identity_field.gaussian_embeddings = self.gauss_params["identity_embeddings"]
            identity_losses = compute_identity_training_loss(
                self.identity_field,
                pixel_embeddings,
                target_labels,
                self.means,
                ignore_index=self.config.identity_ignore_index,
                class_balance_power=self.config.identity_class_balance_power,
                focal_gamma=self.config.identity_focal_gamma,
                spatial_loss_weight=self.config.identity_spatial_loss_weight,
                spatial_k_neighbors=self.config.identity_spatial_k_neighbors,
                spatial_max_samples=self.config.identity_spatial_max_samples,
            )
            loss_dict["identity_loss"] = self.config.identity_loss_weight * identity_losses.total
            loss_dict["identity_cross_entropy"] = identity_losses.cross_entropy
            loss_dict["identity_spatial_consistency"] = identity_losses.spatial_consistency
            return loss_dict


else:

    @dataclass
    class IdentitySplatfactoModelConfig:  # pragma: no cover - import-only fallback
        """Fallback config used when nerfstudio is unavailable."""

        identity_enabled: bool = False
        identity_embedding_dim: int = 16
        identity_num_classes: int = 64
        identity_ignore_index: int = -1
        identity_loss_weight: float = 1.0
        identity_class_balance_power: float = 0.5
        identity_focal_gamma: float = 0.0
        identity_spatial_loss_weight: float = 0.1
        identity_spatial_k_neighbors: int = 8
        identity_spatial_max_samples: int = 4096
        identity_render_key: str = "render_object"
        identity_only: bool = False


    class IdentitySplatfactoModel:  # pragma: no cover - import-only fallback
        def __init__(self, *args, **kwargs):
            raise ImportError("IdentitySplatfactoModel requires nerfstudio and gsplat to be installed.")
