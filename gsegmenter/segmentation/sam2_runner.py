"""Helpers for launching SAM 2 extraction in a separate Python environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(slots=True)
class Sam2ExtractionSpec:
    """Resolved inputs for a single SAM 2 extraction run."""

    python_bin: str
    script_path: Path
    images_dir: Path
    output_root: Path
    checkpoint_path: Path
    model_config: str
    limit: int | None = None
    skip_existing: bool = False
    extra_args: tuple[str, ...] = ()


def build_sam2_extract_command(spec: Sam2ExtractionSpec) -> list[str]:
    """Build the argv list for a SAM 2 extraction subprocess."""

    command = [
        spec.python_bin,
        str(spec.script_path),
        "--images-dir",
        str(spec.images_dir),
        "--output-root",
        str(spec.output_root),
        "--checkpoint-path",
        str(spec.checkpoint_path),
        "--model-config",
        spec.model_config,
    ]
    if spec.limit is not None:
        command.extend(["--limit", str(spec.limit)])
    if spec.skip_existing:
        command.append("--skip-existing")
    command.extend(spec.extra_args)
    return command


def validate_sam2_spec(spec: Sam2ExtractionSpec) -> None:
    """Fail fast when the external SAM 2 launcher inputs are missing."""

    if not spec.script_path.exists():
        raise FileNotFoundError(f"SAM 2 extraction script does not exist: {spec.script_path}")
    if not spec.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {spec.images_dir}")
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint does not exist: {spec.checkpoint_path}")


def run_sam2_extraction(
    spec: Sam2ExtractionSpec,
    *,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | list[str]:
    """Run or preview the external SAM 2 extraction command."""

    command = build_sam2_extract_command(spec)
    if dry_run:
        return command

    validate_sam2_spec(spec)
    spec.output_root.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, check=True, text=True)
