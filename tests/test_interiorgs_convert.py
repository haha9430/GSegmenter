from __future__ import annotations

import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gsegmenter.data import (
    InteriorGSConvertSpec,
    build_interiorgs_convert_command,
    resolve_splat_transform_bin,
    validate_interiorgs_convert_spec,
)


def test_build_interiorgs_convert_command() -> None:
    spec = InteriorGSConvertSpec(
        input_path=Path("scene/3dgs_compressed.ply"),
        output_path=Path("scene/3dgs_uncompressed.ply"),
        splat_transform_bin="splat-transform",
        overwrite=True,
        extra_args=("--summary",),
    )

    with patch("gsegmenter.data.interiorgs_convert.shutil.which", return_value=r"C:\Tools\splat-transform.cmd"):
        command = build_interiorgs_convert_command(spec)

    assert command == [
        "cmd",
        "/c",
        r"C:\Tools\splat-transform.cmd",
        str(Path("scene/3dgs_compressed.ply")),
        "--overwrite",
        "--summary",
        str(Path("scene/3dgs_uncompressed.ply")),
    ]


def test_resolve_splat_transform_bin_tries_cmd_suffix() -> None:
    def _which(name: str) -> str | None:
        if name == "splat-transform":
            return None
        if name == "splat-transform.cmd":
            return r"C:\Users\test\AppData\Roaming\npm\splat-transform.cmd"
        return None

    with patch("gsegmenter.data.interiorgs_convert.shutil.which", side_effect=_which):
        resolved = resolve_splat_transform_bin("splat-transform")

    assert resolved.endswith("splat-transform.cmd")


def test_validate_interiorgs_convert_spec_requires_existing_input(tmp_path: Path) -> None:
    spec = InteriorGSConvertSpec(
        input_path=tmp_path / "missing.ply",
        output_path=tmp_path / "out.ply",
    )

    try:
        validate_interiorgs_convert_spec(spec)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected validation failure for missing input path.")

    assert "does not exist" in message


def test_validate_interiorgs_convert_spec_accepts_existing_ply(tmp_path: Path) -> None:
    input_path = tmp_path / "3dgs_compressed.ply"
    input_path.write_bytes(b"ply\n")
    spec = InteriorGSConvertSpec(
        input_path=input_path,
        output_path=tmp_path / "out.ply",
    )

    validate_interiorgs_convert_spec(spec)
