"""NerfStudio Splatfacto training helpers.

This module keeps the NerfStudio integration thin and explicit so the repo can
stay compatible with future Splatfacto or gsplat changes without coupling core
code to a particular CLI layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path

from gsegmenter.conf import AppConfig


@dataclass(slots=True)
class SplatfactoTrainingSpec:
    """Resolved inputs for a single NerfStudio training run.

    Attributes:
        data_path: Dataset root passed to NerfStudio.
        output_dir: Directory where NerfStudio should write checkpoints/logs.
        num_iterations: Training length in iterations.
        seed: Random seed forwarded to NerfStudio's machine config.
        eval_interval: Evaluation frequency for batch/image evaluation.
        save_interval: Checkpoint save frequency.
        mixed_precision: Whether to enable mixed precision training.
        ns_train_bin: Path or command name used to launch NerfStudio.
        data_parser: Optional NerfStudio data parser subcommand such as
            ``colmap``. When set, dataset-specific arguments should be routed
            through that subcommand instead of the generic top-level alias.
        extra_args: Additional CLI flags passed through unchanged.
    """

    data_path: Path
    output_dir: Path
    num_iterations: int = 30_000
    seed: int = 42
    eval_interval: int = 500
    save_interval: int = 1_000
    mixed_precision: bool = False
    ns_train_bin: str = "ns-train"
    data_parser: str | None = None
    extra_args: tuple[str, ...] = ()


def resolve_splatfacto_spec(config: AppConfig) -> SplatfactoTrainingSpec:
    """Resolve the training spec from the top-level project config."""

    return SplatfactoTrainingSpec(
        data_path=config.dataset.scene_root,
        output_dir=config.training.output_root / config.dataset.scene_name,
        num_iterations=config.training.num_iterations,
        seed=config.training.seed,
        eval_interval=config.training.eval_interval,
        save_interval=config.training.save_interval,
        mixed_precision=config.training.mixed_precision,
    )


def validate_training_spec(spec: SplatfactoTrainingSpec) -> None:
    """Fail fast when required training inputs are missing.

    NerfStudio expects a prepared dataset directory. This check keeps the error
    message close to the wrapper instead of letting the launch fail deep inside
    the training stack.
    """

    if not spec.data_path.exists():
        raise FileNotFoundError(
            f"NerfStudio data directory does not exist: {spec.data_path}. "
            "Pass --data-path to a prepared dataset root or create the expected "
            "scene directory before training."
        )


def build_ns_train_command(spec: SplatfactoTrainingSpec) -> list[str]:
    """Build the `ns-train` command for Splatfacto.

    The command is intentionally returned as a tokenized argv list so the caller
    can execute it safely without shell interpolation.
    """

    command = [
        spec.ns_train_bin,
        "splatfacto",
        "--output-dir",
        str(spec.output_dir),
        "--machine.seed",
        str(spec.seed),
        "--steps-per-save",
        str(spec.save_interval),
        "--steps-per-eval-batch",
        str(spec.eval_interval),
        "--steps-per-eval-image",
        str(spec.eval_interval),
        "--steps-per-eval-all-images",
        str(max(spec.eval_interval * 50, spec.eval_interval)),
        "--max-num-iterations",
        str(spec.num_iterations),
        "--mixed-precision",
        "True" if spec.mixed_precision else "False",
    ]
    if spec.data_parser:
        command.extend([spec.data_parser, "--data", str(spec.data_path)])
    else:
        command[2:2] = ["--data", str(spec.data_path)]
    command.extend(spec.extra_args)
    return command


def run_splatfacto_training(
    spec: SplatfactoTrainingSpec,
    *,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | list[str]:
    """Run or preview a Splatfacto training invocation."""

    command = build_ns_train_command(spec)
    if dry_run:
        return command

    validate_training_spec(spec)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, check=True, text=True)
