"""Export Gaussian-level identity segmentation from an identity-aware checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from gsegmenter.training.identity_eval import load_identity_eval_setup
from gsegmenter.training.identity_export import (
    apply_identity_colors_to_tensors,
    build_identity_export_tensors,
    classify_gaussian_identities,
    filter_finite_and_visible_gaussians,
    identity_palette_rgb,
)
from scripts.export_identity_splat import write_binary_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify each Gaussian identity embedding and export a class-colored PLY."
    )
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-filename", type=str, default="identity_colored_splat.ply")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional identity_mask_manifest.json with raw class names.",
    )
    parser.add_argument(
        "--write-rgb-ply",
        action="store_true",
        help="Also write identity_colored_rgb.ply using red/green/blue uchar properties.",
    )
    parser.add_argument(
        "--include-classes",
        type=str,
        nargs="*",
        default=None,
        help="Only color these class names or numeric class ids. Others keep original colors.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only color Gaussians whose max class probability is at least this value.",
    )
    parser.add_argument(
        "--dim-excluded-opacity-scale",
        type=float,
        default=1.0,
        help="Scale opacity logits of non-highlighted Gaussians. Values below 1.0 dim context.",
    )
    return parser.parse_args()


def _build_class_names(metadata_json: Path | None, class_count: int) -> list[str]:
    names = [f"class_{index}" for index in range(class_count)]
    if metadata_json is None or not metadata_json.exists():
        return names
    payload = json.loads(metadata_json.read_text(encoding="utf-8"))
    for entry in payload.get("classes", []):
        class_id = int(entry["global_id"])
        if 0 <= class_id < class_count:
            names[class_id] = str(entry["raw_key"])
    return names


def _build_rgb_tensors(base_tensors, identity_ids: np.ndarray, palette_rgb: np.ndarray):
    rgb = np.clip(palette_rgb[identity_ids], 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
    tensors = base_tensors.copy()
    for key in list(tensors.keys()):
        if key.startswith("f_dc_") or key.startswith("f_rest_"):
            del tensors[key]
    tensors["red"] = rgb_u8[:, 0]
    tensors["green"] = rgb_u8[:, 1]
    tensors["blue"] = rgb_u8[:, 2]
    return tensors


def _resolve_include_class_ids(include_classes: list[str] | None, class_names: list[str]) -> set[int] | None:
    if include_classes is None:
        return None
    by_name = {name.casefold(): index for index, name in enumerate(class_names)}
    include_ids: set[int] = set()
    for raw_value in include_classes:
        value = str(raw_value).strip()
        if value.isdigit():
            include_ids.add(int(value))
            continue
        lowered = value.casefold()
        if lowered not in by_name:
            raise ValueError(
                f"Unknown class {value!r}. Available classes: {', '.join(class_names)}"
            )
        include_ids.add(by_name[lowered])
    return include_ids


def _apply_partial_identity_colors(
    base_tensors,
    identity_ids: np.ndarray,
    probabilities: np.ndarray,
    *,
    palette_rgb: np.ndarray,
    include_class_ids: set[int] | None,
    min_confidence: float,
    dim_excluded_opacity_scale: float,
):
    colored = base_tensors.copy()
    confidence = probabilities[np.arange(probabilities.shape[0]), identity_ids]
    selection = confidence >= float(min_confidence)
    if include_class_ids is not None:
        selection &= np.isin(identity_ids, np.asarray(sorted(include_class_ids), dtype=np.int32))

    if np.any(selection):
        selected_tensors = apply_identity_colors_to_tensors(
            base_tensors,
            identity_ids,
            palette_rgb=palette_rgb,
        )
        for key in ("f_dc_0", "f_dc_1", "f_dc_2"):
            colored[key][selection] = selected_tensors[key][selection]
        for key in colored:
            if key.startswith("f_rest_"):
                colored[key][selection] = 0.0
    if dim_excluded_opacity_scale != 1.0:
        if dim_excluded_opacity_scale <= 0.0:
            raise ValueError("--dim-excluded-opacity-scale must be positive.")
        non_selected = ~selection
        colored["opacity"][non_selected] = colored["opacity"][non_selected] + np.float32(
            np.log(float(dim_excluded_opacity_scale))
        )
    return colored, selection, confidence


def main() -> int:
    args = parse_args()
    _, pipeline, checkpoint_path, step = load_identity_eval_setup(args.load_config, test_mode="inference")
    model = pipeline.model
    identity_embeddings = getattr(model, "identity_embeddings", None)
    identity_field = getattr(model, "identity_field", None)
    if identity_embeddings is None or identity_field is None:
        raise ValueError("Loaded model does not expose identity embeddings and identity_field.")

    identity_ids, probabilities = classify_gaussian_identities(
        identity_embeddings=identity_embeddings,
        classifier=identity_field.classifier,
    )
    shs_rest = None
    if hasattr(model, "shs_rest"):
        shs_rest = model.shs_rest.detach().transpose(1, 2).contiguous().cpu().numpy()
    tensors = build_identity_export_tensors(
        positions=model.means.detach().cpu().numpy(),
        opacities=model.opacities.detach().cpu().numpy(),
        scales=model.scales.detach().cpu().numpy(),
        quats=model.quats.detach().cpu().numpy(),
        shs_0=model.shs_0.detach().contiguous().cpu().numpy(),
        shs_rest=shs_rest,
        ply_color_mode="sh_coeffs",
    )
    filtered_tensors, invalid_count, low_opacity_count, keep_mask = filter_finite_and_visible_gaussians(tensors)
    filtered_identity_ids = identity_ids[keep_mask]
    filtered_probabilities = probabilities[keep_mask]
    class_count = int(probabilities.shape[1])
    palette = identity_palette_rgb(class_count)
    class_names = _build_class_names(args.metadata_json, class_count)
    include_class_ids = _resolve_include_class_ids(args.include_classes, class_names)
    colored_tensors, highlight_mask, confidence = _apply_partial_identity_colors(
        filtered_tensors,
        filtered_identity_ids,
        filtered_probabilities,
        palette_rgb=palette,
        include_class_ids=include_class_ids,
        min_confidence=args.min_confidence,
        dim_excluded_opacity_scale=args.dim_excluded_opacity_scale,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_filename
    write_binary_ply(output_path, colored_tensors)
    np.save(args.output_dir / "gaussian_identity_ids.npy", filtered_identity_ids.astype(np.int32))
    np.save(args.output_dir / "gaussian_identity_probabilities.npy", filtered_probabilities.astype(np.float32))
    np.save(args.output_dir / "gaussian_identity_highlight_mask.npy", highlight_mask.astype(bool))
    np.save(args.output_dir / "gaussian_identity_confidence.npy", confidence.astype(np.float32))
    np.save(args.output_dir / "export_keep_mask.npy", keep_mask)

    class_counts = np.bincount(filtered_identity_ids, minlength=class_count)
    highlight_counts = np.bincount(filtered_identity_ids[highlight_mask], minlength=class_count)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "step": int(step),
        "gaussian_count": int(filtered_identity_ids.shape[0]),
        "invalid_filtered_count": int(invalid_count),
        "low_opacity_filtered_count": int(low_opacity_count),
        "highlighted_gaussian_count": int(np.count_nonzero(highlight_mask)),
        "min_confidence": float(args.min_confidence),
        "include_classes": None if include_class_ids is None else sorted(int(value) for value in include_class_ids),
        "dim_excluded_opacity_scale": float(args.dim_excluded_opacity_scale),
        "classes": [
            {
                "class_id": class_id,
                "name": class_names[class_id],
                "gaussian_count": int(class_counts[class_id]),
                "highlighted_gaussian_count": int(highlight_counts[class_id]),
                "rgb": palette[class_id].tolist(),
            }
            for class_id in range(class_count)
        ],
        "outputs": {
            "identity_colored_ply": str(output_path),
            "gaussian_identity_ids": str(args.output_dir / "gaussian_identity_ids.npy"),
            "gaussian_identity_probabilities": str(args.output_dir / "gaussian_identity_probabilities.npy"),
            "gaussian_identity_highlight_mask": str(args.output_dir / "gaussian_identity_highlight_mask.npy"),
            "gaussian_identity_confidence": str(args.output_dir / "gaussian_identity_confidence.npy"),
            "export_keep_mask": str(args.output_dir / "export_keep_mask.npy"),
        },
    }
    if args.write_rgb_ply:
        rgb_path = args.output_dir / "identity_colored_rgb.ply"
        write_binary_ply(rgb_path, _build_rgb_tensors(filtered_tensors, filtered_identity_ids, palette))
        summary["outputs"]["identity_colored_rgb_ply"] = str(rgb_path)

    summary_path = args.output_dir / "identity_classes.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded checkpoint: {checkpoint_path} (step {step})")
    print(f"Exported identity-colored PLY to {output_path}")
    print(f"Wrote Gaussian identity IDs to {args.output_dir / 'gaussian_identity_ids.npy'}")
    print(f"Wrote class summary to {summary_path}")
    print(f"Highlighted {summary['highlighted_gaussian_count']} / {summary['gaussian_count']} gaussians")
    for entry in summary["classes"]:
        print(
            f"  class={entry['class_id']} name={entry['name']} "
            f"gaussians={entry['gaussian_count']} highlighted={entry['highlighted_gaussian_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
