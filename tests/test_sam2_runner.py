from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.segmentation.sam2_runner import (  # noqa: E402
    Sam2ExtractionSpec,
    build_sam2_extract_command,
    run_sam2_extraction,
    validate_sam2_spec,
)


def test_build_sam2_extract_command_includes_expected_flags():
    spec = Sam2ExtractionSpec(
        python_bin=r"C:\envs\sam2\python.exe",
        script_path=Path("scripts/extract_sam2_masks.py"),
        images_dir=Path("data/library/images"),
        output_root=Path("outputs/library/masks"),
        checkpoint_path=Path("checkpoints/sam2.pt"),
        model_config="sam2_hiera_l.yaml",
        limit=10,
        skip_existing=True,
        extra_args=("--foo", "bar"),
    )

    command = build_sam2_extract_command(spec)

    assert command[:2] == [
        r"C:\envs\sam2\python.exe",
        str(Path("scripts") / "extract_sam2_masks.py"),
    ]
    assert "--images-dir" in command
    assert str(Path("data") / "library" / "images") in command
    assert "--output-root" in command
    assert str(Path("outputs") / "library" / "masks") in command
    assert "--checkpoint-path" in command
    assert str(Path("checkpoints") / "sam2.pt") in command
    assert "--model-config" in command
    assert "sam2_hiera_l.yaml" in command
    assert "--limit" in command
    assert "10" in command
    assert "--skip-existing" in command
    assert command[-2:] == ["--foo", "bar"]


def test_run_sam2_extraction_dry_run_uses_external_python():
    spec = Sam2ExtractionSpec(
        python_bin="python-sam2",
        script_path=Path("scripts/extract_sam2_masks.py"),
        images_dir=Path("data/images"),
        output_root=Path("outputs/masks"),
        checkpoint_path=Path("checkpoints/sam2.pt"),
        model_config="cfg.yaml",
    )

    command = run_sam2_extraction(spec, dry_run=True)

    assert command[0] == "python-sam2"


def test_validate_sam2_spec_raises_for_missing_inputs(tmp_path):
    spec = Sam2ExtractionSpec(
        python_bin="python",
        script_path=tmp_path / "missing_script.py",
        images_dir=tmp_path / "images",
        output_root=tmp_path / "outputs",
        checkpoint_path=tmp_path / "missing_checkpoint.pt",
        model_config="cfg.yaml",
    )

    try:
        validate_sam2_spec(spec)
    except FileNotFoundError as exc:
        assert "SAM 2 extraction script does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing script path")
