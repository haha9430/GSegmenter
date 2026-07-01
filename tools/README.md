# Tools

This folder is reserved for utility scripts such as mask extraction, Gaussian
group building, and data-format conversion helpers.

Current utilities:

- `visualize_groups.py`: export grouped Gaussians as a colored point cloud PLY.
  Use `--show-context --highlight-red` to keep the full 3D scene in gray while
  highlighting selected groups in red.
- `visualize_group_projection.py`: overlay top Gaussian groups on a source frame.
  By default it now renders high-visibility red boundaries; add `--show-points`
  if you also want the projected support points.
- `highlight_group_ply.py`: export a new `splat.ply` copy that preserves the
  original 3DGS colors while recoloring selected groups through their SH DC
  channels. Use `--multi-color` to color each selected object id differently,
  `--include-labels chair table` to select InteriorGS-style labels directly,
  `--dim-opacity-scale` to suppress background noise, and
  `--flatten-selected-sh` when the viewer's higher-order SH makes highlights hard
  to read.
- `export_group_best_frames.py`: find the frame where each selected group is most
  visible and export a boundary overlay image plus a JSON summary. This is the
  fastest way to understand what a numeric object id corresponds to.
- `render_identity_preview.py`: load an `identity-splatfacto` config/checkpoint
  and export a side-by-side PNG with RGB, predicted identity colors, predicted
  overlay, and GT identity labels when available.
- `..\scripts\export_identity_splat.py`: bypass `ns-export` and write a regular
  `splat.ply` directly from an `identity-splatfacto` config/checkpoint. Use
  `--write-identity-sidecar` if you also want `identity_embeddings.npy`.
- `visualize_group_comparison.py`: color a Gaussian PLY by GT/pred agreement.
  The output uses green for matched Gaussians, red for GT-only misses, blue for
  pred-only false positives, yellow for conflicting assignments, and dim gray
  for background. This is the fastest way to debug SAM 2 lifting quality once a
  predicted `gaussian_object_ids.npy` is available.
