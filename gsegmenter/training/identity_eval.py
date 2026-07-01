"""Evaluation helpers for identity-aware checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import yaml

try:  # pragma: no cover - exercised when nerfstudio is available.
    from nerfstudio.engine.trainer import TrainerConfig
    from nerfstudio.utils.eval_utils import eval_load_checkpoint

    HAS_NERFSTUDIO_EVAL = True
except ImportError:  # pragma: no cover - import-only fallback
    TrainerConfig = object  # type: ignore[assignment]
    eval_load_checkpoint = None
    HAS_NERFSTUDIO_EVAL = False


def load_identity_eval_setup(
    config_path: Path,
    *,
    test_mode: str = "test",
) -> Tuple["TrainerConfig", object, Path, int]:
    """Load a local identity-aware pipeline without relying on Nerfstudio method registry."""

    if not HAS_NERFSTUDIO_EVAL:
        raise ImportError("Identity eval setup requires nerfstudio to be installed.")

    config = yaml.load(Path(config_path).read_text(encoding="utf-8"), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.load_dir = config.get_checkpoint_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    pipeline.eval()
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)
    return config, pipeline, checkpoint_path, step
