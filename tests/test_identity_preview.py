from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.render_identity_preview import blend_overlay, colorize_label_map, palette_color


def test_palette_color_is_deterministic() -> None:
    assert palette_color(3) == palette_color(3)
    assert palette_color(-1) == (0, 0, 0)


def test_colorize_label_map_assigns_colors() -> None:
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    colored = colorize_label_map(labels)

    assert colored.shape == (2, 2, 3)
    assert tuple(colored[0, 0]) == palette_color(0)
    assert tuple(colored[0, 1]) == palette_color(1)
    assert tuple(colored[1, 1]) == palette_color(2)


def test_blend_overlay_preserves_shape() -> None:
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    identity = np.full((2, 2, 3), 200, dtype=np.uint8)

    blended = blend_overlay(rgb, identity, alpha=0.5)

    assert blended.shape == rgb.shape
    assert blended.dtype == np.uint8
    assert int(blended[0, 0, 0]) == 150
