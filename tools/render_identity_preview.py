"""Render RGB and identity maps from an identity-aware Splatfacto checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from gsegmenter.training.identity_eval import load_identity_eval_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an identity preview from a trained checkpoint.")
    parser.add_argument("--load-config", type=Path, required=True, help="Path to an identity-splatfacto config.yml")
    parser.add_argument("--frame-index", type=int, default=0, help="Eval frame index to render.")
    parser.add_argument("--output-path", type=Path, required=True, help="PNG path for the preview contact sheet.")
    parser.add_argument("--overlay-alpha", type=float, default=0.55, help="Blend weight for identity overlay.")
    parser.add_argument("--draw-grid", action="store_true", help="Draw panel labels on the contact sheet.")
    return parser.parse_args()


def palette_color(index: int) -> tuple[int, int, int]:
    """Deterministic bright palette that stays readable over RGB renders."""

    if index < 0:
        return (0, 0, 0)
    hue = (index * 0.61803398875) % 1.0
    r, g, b = hsv_to_rgb(hue, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    """Convert a float RGB tensor in `[0, 1]` to uint8 image memory."""

    image = image.detach().cpu().clamp(0.0, 1.0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Turn an integer label map into a colorful uint8 RGB image."""

    colored = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(label_map)
    for label in unique_labels:
        colored[label_map == label] = palette_color(int(label))
    return colored


def blend_overlay(rgb: np.ndarray, identity_rgb: np.ndarray, *, alpha: float) -> np.ndarray:
    """Blend an identity map over an RGB render for quick visual inspection."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = rgb.astype(np.float32) * (1.0 - alpha) + identity_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def add_panel_titles(images: list[Image.Image], titles: list[str]) -> list[Image.Image]:
    labeled: list[Image.Image] = []
    for image, title in zip(images, titles):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
        draw.text((8, 5), title, fill=(255, 255, 255))
        labeled.append(canvas)
    return labeled


def main() -> int:
    args = parse_args()
    _, pipeline, checkpoint_path, step = load_identity_eval_setup(args.load_config, test_mode="test")
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader
    if args.frame_index < 0 or args.frame_index >= len(dataloader):
        raise IndexError(f"frame_index {args.frame_index} is outside eval range [0, {len(dataloader) - 1}]")

    camera, batch = dataloader[args.frame_index]
    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)

    rgb = tensor_to_uint8_image(outputs["rgb"])
    render_object = outputs["render_object"]
    if render_object.ndim != 3:
        raise ValueError(f"Expected render_object with shape `(H, W, D)`, got {tuple(render_object.shape)}")

    pixel_embeddings = render_object.permute(2, 0, 1).unsqueeze(0)
    logits = pipeline.model.identity_field.classifier(pixel_embeddings).squeeze(0)
    predicted_labels = torch.argmax(logits, dim=0).detach().cpu().numpy().astype(np.int32)
    predicted_color = colorize_label_map(predicted_labels)
    predicted_overlay = blend_overlay(rgb, predicted_color, alpha=args.overlay_alpha)

    panels = [
        Image.fromarray(rgb, mode="RGB"),
        Image.fromarray(predicted_color, mode="RGB"),
        Image.fromarray(predicted_overlay, mode="RGB"),
    ]
    titles = ["RGB", "Pred Identity", "Pred Overlay"]

    if "identity_labels" in batch:
        gt_labels = batch["identity_labels"].detach().cpu().numpy().astype(np.int32)
        gt_color = colorize_label_map(gt_labels)
        gt_overlay = blend_overlay(rgb, gt_color, alpha=args.overlay_alpha)
        panels.extend(
            [
                Image.fromarray(gt_color, mode="RGB"),
                Image.fromarray(gt_overlay, mode="RGB"),
            ]
        )
        titles.extend(["GT Identity", "GT Overlay"])

    if args.draw_grid:
        panels = add_panel_titles(panels, titles)

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height))
    cursor_x = 0
    for panel in panels:
        sheet.paste(panel, (cursor_x, 0))
        cursor_x += panel.width

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output_path)
    print(f"Loaded checkpoint: {checkpoint_path} (step {step})")
    print(f"Wrote identity preview to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
