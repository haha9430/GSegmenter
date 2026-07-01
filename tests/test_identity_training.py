from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.segmentation.mask_io import (
    FrameMasksManifest,
    MaskInstanceRecord,
    save_binary_mask,
    save_frame_masks_manifest,
)
from gsegmenter.training.identity_bridge import (
    build_identity_training_batch,
    extract_rendered_identity_embeddings,
)
from gsegmenter.training.identity_datamanager import (
    build_identity_frame_lookup,
    normalize_identity_image_path,
    prepare_identity_label_map,
    resolve_identity_frame,
)
from gsegmenter.training.identity_dataset import SceneIdentityLabelFrame, load_scene_identity_frames
from gsegmenter.training.identity_loss import compute_identity_training_loss
from gsegmenter.training.identity_loss import compute_balanced_class_weights
from gsegmenter.training.identity_method import build_identity_optimizer_config
from gsegmenter.training.identity_method import infer_identity_num_classes
from gsegmenter.training.identity_runner import IdentitySplatfactoTrainingSpec, build_identity_trainer_config
from gsegmenter.training.identity_step import run_identity_optimization_step
from gsegmenter.training.identity_splatfacto import (
    IdentitySplatfactoModelConfig,
    prepare_identity_label_tensor,
)
from gsegmenter.training.identity_supervision import rasterize_identity_labels
from gsegmenter.training.object_field import GaussianIdentityField
from gsegmenter.training.regularization import identity_spatial_consistency_loss


def test_rasterize_identity_labels_prefers_higher_scores(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frame_00001"
    frame_dir.mkdir(parents=True)
    low_score_mask = np.zeros((4, 4), dtype=bool)
    low_score_mask[1:3, 1:3] = True
    high_score_mask = np.zeros((4, 4), dtype=bool)
    high_score_mask[2:, 2:] = True
    save_binary_mask(low_score_mask, frame_dir / "mask_low.png")
    save_binary_mask(high_score_mask, frame_dir / "mask_high.png")

    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="images/frame.png",
        image_size=(4, 4),
        instances=(
            MaskInstanceRecord(7, (1, 1, 2, 2), 0.2, 4, "mask_low.png"),
            MaskInstanceRecord(9, (2, 2, 3, 3), 0.9, 4, "mask_high.png"),
        ),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")

    labels = rasterize_identity_labels(manifest, frame_dir, min_score=0.0)

    assert labels.class_ids == (7, 9)
    assert labels.label_map.shape == (4, 4)
    assert labels.label_map[2, 2] == 1
    assert labels.label_map[1, 1] == 0
    assert labels.label_map[0, 0] == -1


def test_gaussian_identity_field_shapes() -> None:
    field = GaussianIdentityField(num_gaussians=5, embedding_dim=8, num_classes=3)
    pixel_embeddings = torch.randn(2, 8, 16, 16)

    output = field(pixel_embeddings)

    assert output.gaussian_embeddings.shape == (5, 8)
    assert output.pixel_embeddings.shape == (2, 8, 16, 16)
    assert output.pixel_logits.shape == (2, 3, 16, 16)


def test_scene_identity_vocabulary_remaps_consistently(tmp_path: Path) -> None:
    frame_a = tmp_path / "frame_00001"
    frame_b = tmp_path / "frame_00002"
    frame_a.mkdir(parents=True)
    frame_b.mkdir(parents=True)

    mask_a = np.zeros((3, 3), dtype=bool)
    mask_a[:2, :2] = True
    mask_b = np.zeros((3, 3), dtype=bool)
    mask_b[1:, 1:] = True
    save_binary_mask(mask_a, frame_a / "mask_a.png")
    save_binary_mask(mask_b, frame_b / "mask_b.png")

    manifest_a = FrameMasksManifest(
        frame_index=0,
        image_path="images/a.png",
        image_size=(3, 3),
        instances=(MaskInstanceRecord(42, (0, 0, 1, 1), 0.9, 4, "mask_a.png"),),
    )
    manifest_b = FrameMasksManifest(
        frame_index=1,
        image_path="images/b.png",
        image_size=(3, 3),
        instances=(MaskInstanceRecord(42, (1, 1, 2, 2), 0.8, 4, "mask_b.png"),),
    )
    save_frame_masks_manifest(manifest_a, frame_a / "instances.json")
    save_frame_masks_manifest(manifest_b, frame_b / "instances.json")

    vocabulary, frames = load_scene_identity_frames(tmp_path)

    assert vocabulary.raw_object_ids == (42,)
    assert vocabulary.num_classes == 1
    assert len(frames) == 2
    assert np.all(frames[0].label_map[mask_a] == 0)
    assert np.all(frames[1].label_map[mask_b] == 0)


def test_compute_identity_training_loss_shapes() -> None:
    field = GaussianIdentityField(num_gaussians=4, embedding_dim=6, num_classes=3)
    pixel_embeddings = torch.randn(2, 6, 5, 5)
    target_labels = torch.full((2, 5, 5), -1, dtype=torch.int64)
    target_labels[:, 1:4, 1:4] = 1
    xyz = torch.randn(4, 3)

    breakdown = compute_identity_training_loss(
        field,
        pixel_embeddings,
        target_labels,
        xyz,
        ignore_index=-1,
        class_balance_power=0.5,
        focal_gamma=0.0,
        spatial_loss_weight=0.25,
        spatial_k_neighbors=2,
        spatial_max_samples=4,
    )

    assert breakdown.total.ndim == 0
    assert breakdown.cross_entropy.ndim == 0
    assert breakdown.spatial_consistency.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert breakdown.valid_pixel_count == 18


def test_compute_balanced_class_weights_boosts_rare_labels() -> None:
    labels = torch.tensor(
        [
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=torch.int64,
    )
    weights = compute_balanced_class_weights(
        labels,
        num_classes=3,
        ignore_index=-1,
        balance_power=1.0,
    )

    assert weights.shape[0] == 3
    assert weights[1] > weights[0]
    assert weights[2] == 1.0


def test_build_identity_training_batch_from_renderer_outputs(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frame_00001"
    frame_dir.mkdir(parents=True)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    save_binary_mask(mask, frame_dir / "mask.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="images/frame.png",
        image_size=(4, 4),
        instances=(MaskInstanceRecord(3, (1, 1, 2, 2), 0.9, 4, "mask.png"),),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")
    _, frames = load_scene_identity_frames(tmp_path)

    renderer_outputs = {"render_object": torch.randn(1, 5, 4, 4)}
    gaussian_xyz = torch.randn(6, 3)
    batch = build_identity_training_batch(renderer_outputs, frames, [0], gaussian_xyz)

    assert batch.frame_indices == (0,)
    assert batch.pixel_embeddings.shape == (1, 5, 4, 4)
    assert batch.target_labels.shape == (1, 4, 4)
    assert batch.gaussian_xyz.shape == (6, 3)


def test_extract_rendered_identity_embeddings_accepts_single_frame() -> None:
    outputs = {"render_object": torch.randn(6, 8, 8)}
    embeddings = extract_rendered_identity_embeddings(outputs)
    assert embeddings.shape == (1, 6, 8, 8)


def test_prepare_identity_label_tensor_resizes_nearest() -> None:
    labels = torch.tensor(
        [
            [0, 0, -1, -1],
            [0, 1, 1, -1],
            [2, 2, 1, -1],
            [2, 2, 2, -1],
        ],
        dtype=torch.int64,
    )
    resized = prepare_identity_label_tensor(labels, (2, 2), ignore_index=-1)
    assert resized.shape == (1, 2, 2)
    assert resized[0, 0, 0].item() == 0
    assert resized[0, 0, 1].item() == -1


def test_identity_splatfacto_config_defaults_import_without_nerfstudio() -> None:
    config = IdentitySplatfactoModelConfig()
    assert config.identity_render_key == "render_object"


def test_identity_lookup_resolves_by_path_and_basename() -> None:
    frames = (
        SceneIdentityLabelFrame(
            frame_index=3,
            image_path="images/frame_00003.png",
            label_map=np.zeros((2, 2), dtype=np.int32),
            ignore_index=-1,
        ),
    )
    by_path, by_index = build_identity_frame_lookup(frames)

    resolved_direct = resolve_identity_frame("images/frame_00003.png", 99, by_path=by_path, by_index=by_index)
    resolved_basename = resolve_identity_frame("C:/tmp/frame_00003.png", 99, by_path=by_path, by_index=by_index)
    resolved_index = resolve_identity_frame("images/missing.png", 3, by_path=by_path, by_index=by_index)

    assert normalize_identity_image_path("./images/frame_00003.png") == "images/frame_00003.png"
    assert resolved_direct is frames[0]
    assert resolved_basename is frames[0]
    assert resolved_index is frames[0]


def test_prepare_identity_label_map_resizes_to_cached_image_shape() -> None:
    label_map = np.array(
        [
            [0, 0, -1, -1],
            [0, 1, 1, -1],
            [2, 2, 1, -1],
            [2, 2, 2, -1],
        ],
        dtype=np.int32,
    )

    prepared = prepare_identity_label_map(label_map, (2, 2), ignore_index=-1)

    assert prepared.shape == (2, 2)
    assert prepared[0, 0].item() == 0
    assert prepared[0, 1].item() == -1


def test_identity_optimizer_config_includes_new_param_groups() -> None:
    optimizers = build_identity_optimizer_config()
    if optimizers:
        assert "identity_embeddings" in optimizers
        assert "identity_field" in optimizers


def test_identity_only_optimizer_config_excludes_geometry_groups() -> None:
    optimizers = build_identity_optimizer_config(identity_only=True)
    if optimizers:
        assert set(optimizers) == {"identity_embeddings", "identity_field"}


def test_infer_identity_num_classes_counts_scene_global_ids(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frame_00001"
    frame_dir.mkdir(parents=True)
    mask_a = np.zeros((2, 2), dtype=bool)
    mask_a[:, 0] = True
    mask_b = np.zeros((2, 2), dtype=bool)
    mask_b[:, 1] = True
    save_binary_mask(mask_a, frame_dir / "mask_a.png")
    save_binary_mask(mask_b, frame_dir / "mask_b.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="images/frame.png",
        image_size=(2, 2),
        instances=(
            MaskInstanceRecord(7, (0, 0, 0, 1), 0.9, 2, "mask_a.png"),
            MaskInstanceRecord(11, (1, 0, 1, 1), 0.9, 2, "mask_b.png"),
        ),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")

    assert infer_identity_num_classes(tmp_path, min_mask_score=0.0) == 2


def test_build_identity_trainer_config_dry_path(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene01"
    scene_root.mkdir(parents=True)
    masks_root = tmp_path / "masks"
    frame_dir = masks_root / "frame_00001"
    frame_dir.mkdir(parents=True)
    mask = np.ones((2, 2), dtype=bool)
    save_binary_mask(mask, frame_dir / "mask.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="images/frame.png",
        image_size=(2, 2),
        instances=(MaskInstanceRecord(5, (0, 0, 1, 1), 0.9, 4, "mask.png"),),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")

    spec = IdentitySplatfactoTrainingSpec(
        data_path=scene_root,
        masks_root=masks_root,
        output_dir=tmp_path / "outputs" / "scene01",
        num_iterations=100,
    )
    if build_identity_optimizer_config():
        config = build_identity_trainer_config(spec)
        assert config.method_name == "identity-splatfacto"
        assert config.pipeline.model.identity_num_classes == 1


def test_build_identity_trainer_config_passes_splatfacto_noise_controls(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene01"
    scene_root.mkdir(parents=True)
    masks_root = tmp_path / "masks"
    frame_dir = masks_root / "frame_00001"
    frame_dir.mkdir(parents=True)
    mask = np.ones((2, 2), dtype=bool)
    save_binary_mask(mask, frame_dir / "mask.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="images/frame.png",
        image_size=(2, 2),
        instances=(MaskInstanceRecord(5, (0, 0, 1, 1), 0.9, 4, "mask.png"),),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")

    spec = IdentitySplatfactoTrainingSpec(
        data_path=scene_root,
        masks_root=masks_root,
        output_dir=tmp_path / "outputs" / "scene01",
        cull_alpha_thresh=0.2,
        cull_scale_thresh=0.25,
        reset_alpha_every=15,
        densify_grad_thresh=0.0012,
        use_scale_regularization=True,
        max_gauss_ratio=4.0,
        identity_min_mask_score=0.25,
    )
    if build_identity_optimizer_config():
        config = build_identity_trainer_config(spec)
        assert config.pipeline.model.cull_alpha_thresh == 0.2
        assert config.pipeline.model.cull_scale_thresh == 0.25
        assert config.pipeline.model.reset_alpha_every == 15
        assert config.pipeline.model.densify_grad_thresh == 0.0012
        assert config.pipeline.model.use_scale_regularization is True
        assert config.pipeline.model.max_gauss_ratio == 4.0
        assert config.pipeline.datamanager.identity_min_mask_score == 0.25


def test_build_identity_trainer_config_identity_only_loads_checkpoint(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene01"
    scene_root.mkdir(parents=True)
    masks_root = tmp_path / "masks"
    frame_dir = masks_root / "frame_00001"
    frame_dir.mkdir(parents=True)
    mask = np.ones((2, 2), dtype=bool)
    save_binary_mask(mask, frame_dir / "mask.png")
    manifest = FrameMasksManifest(
        frame_index=0,
        image_path="images/frame.png",
        image_size=(2, 2),
        instances=(MaskInstanceRecord(5, (0, 0, 1, 1), 0.9, 4, "mask.png"),),
    )
    save_frame_masks_manifest(manifest, frame_dir / "instances.json")
    checkpoint = tmp_path / "baseline.ckpt"
    torch.save(
        {
            "step": 42,
            "pipeline": {},
            "optimizers": {"means": {}},
            "scalers": {},
        },
        checkpoint,
    )

    spec = IdentitySplatfactoTrainingSpec(
        data_path=scene_root,
        masks_root=masks_root,
        output_dir=tmp_path / "outputs" / "scene01",
        identity_only=True,
        load_checkpoint=checkpoint,
        sh_degree=0,
    )
    if build_identity_optimizer_config():
        config = build_identity_trainer_config(spec)
        assert config.pipeline.model.identity_only is True
        assert config.pipeline.model.sh_degree == 0
        assert set(config.optimizers) == {"identity_embeddings", "identity_field"}
        assert config.load_checkpoint.name == "baseline_identity_only.ckpt"
        assert config.load_scheduler is False


def test_run_identity_optimization_step_backpropagates() -> None:
    field = GaussianIdentityField(num_gaussians=4, embedding_dim=6, num_classes=2)
    optimizer = torch.optim.SGD(field.parameters(), lr=0.1)
    pixel_embeddings = torch.randn(1, 6, 4, 4)
    target_labels = torch.zeros((1, 4, 4), dtype=torch.int64)
    gaussian_xyz = torch.randn(4, 3)
    frames = (
        SceneIdentityLabelFrame(
            frame_index=0,
            image_path="images/frame.png",
            label_map=target_labels[0].numpy(),
            ignore_index=-1,
        ),
    )
    batch = build_identity_training_batch(
        {"render_object": pixel_embeddings},
        frames,
        [0],
        gaussian_xyz,
    )

    result = run_identity_optimization_step(
        field,
        batch,
        optimizer=optimizer,
        backward=True,
        spatial_loss_weight=0.1,
        spatial_k_neighbors=2,
    )

    assert result.batch_size == 1
    assert torch.isfinite(result.losses.total)


def test_identity_spatial_consistency_loss_no_nan() -> None:
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss = identity_spatial_consistency_loss(xyz, embeddings, k_neighbors=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_identity_spatial_consistency_loss_samples_large_cloud() -> None:
    xyz = torch.randn(5000, 3)
    embeddings = torch.randn(5000, 4)

    loss = identity_spatial_consistency_loss(xyz, embeddings, k_neighbors=4, max_samples=256)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
