# Scripts

This folder will hold runnable entry points for the NerfStudio-first pipeline.

Planned entry points:

- `run_colmap.py`
- `prepare_video_scene.py`
- `train_splatfacto.py`
- `extract_sam2_masks.py`
- `run_sam2_masks.py`
- `extract_grounded_sam2_masks.py`
- `run_grounded_sam2_masks.py`
- `prepare_identity_masks.py`
- `prepare_single_object_identity_masks.py`
- `prepare_multiclass_category_identity_masks.py`
- `discover_object_proposals.py`
- `tools/discover_category_instances.py` performs training-free first-pass
  instance discovery by lifting grounded masks onto an existing Gaussian PLY.
- `tools/project_discovered_instances_to_masks.py` projects first-pass Gaussian
  instance candidates into frame-wise masks for second-pass identity training.
- `tools/reassign_grounded_masks_to_discovered_instances.py` keeps dense
  Grounded-SAM masks and re-labels them with discovered 3D instance ids.
- `tools/discover_colmap_mask_instances.py` performs PLY-free first-pass
  instance discovery by associating Grounded-SAM masks through COLMAP tracks.
- `train_identity_splatfacto.py` can run full identity-aware Splatfacto, or
  `--identity-only --load-checkpoint <baseline.ckpt>` to freeze an existing
  Gaussian scene and train only identity embeddings/classifier. For
  identity-only runs, pass the same `--sh-degree` used by the baseline
  checkpoint so Gaussian color tensor shapes match.
- `tools/export_identity_instances.py` clusters high-confidence category Gaussians
  from a multi-class identity export into spatial instance candidates.
- `extract_monocular_depths.py`
- `lift_masks_to_gaussians.py`
- `diagnose_gaussian_noise.py`
- `diagnose_gaussian_depth_consistency.py`
- `build_gaussian_quality.py`
- `prune_gaussian_noise.py`
- `inspect_interiorgs_scene.py`
- `convert_interiorgs_ply.py`
- `build_interiorgs_groups.py`
- `apply_group_transform.py`
- `cleanup_group_source_region.py`
- `validate_interiorgs_placement.py`
- `evaluate_gaussian_groups.py`
- `launch_editor.py`
