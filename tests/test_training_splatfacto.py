from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.conf import AppConfig  # noqa: E402
from gsegmenter.training.splatfacto import (  # noqa: E402
    SplatfactoTrainingSpec,
    build_ns_train_command,
    resolve_splatfacto_spec,
    run_splatfacto_training,
    validate_training_spec,
)


def test_resolve_splatfacto_spec_uses_scene_root():
    config = AppConfig()
    config.dataset.data_root = Path("data")
    config.dataset.scene_name = "scene42"
    config.training.output_root = Path("outputs")

    spec = resolve_splatfacto_spec(config)

    assert spec.data_path == Path("data") / "scene42"
    assert spec.output_dir == Path("outputs") / "scene42"


def test_build_ns_train_command_includes_expected_flags():
    spec = SplatfactoTrainingSpec(
        data_path=Path("data/scene01"),
        output_dir=Path("outputs/scene01"),
        num_iterations=123,
        seed=7,
        eval_interval=11,
        save_interval=22,
        mixed_precision=True,
        ns_train_bin=r"C:\Tools\ns-train.exe",
        extra_args=("--viewer.quit-on-train-completion",),
    )

    command = build_ns_train_command(spec)

    assert command[:4] == [r"C:\Tools\ns-train.exe", "splatfacto", "--data", str(Path("data") / "scene01")]
    assert "--output-dir" in command
    assert str(Path("outputs") / "scene01") in command
    assert "--machine.seed" in command
    assert "7" in command
    assert "--steps-per-save" in command
    assert "22" in command
    assert "--steps-per-eval-batch" in command
    assert "11" in command
    assert "--steps-per-eval-image" in command
    assert "--steps-per-eval-all-images" in command
    assert "550" in command
    assert "--max-num-iterations" in command
    assert "123" in command
    assert "--mixed-precision" in command
    assert "True" in command
    assert command[-1] == "--viewer.quit-on-train-completion"


def test_run_splatfacto_training_dry_run_uses_custom_bin():
    spec = SplatfactoTrainingSpec(
        data_path=Path("data/scene01"),
        output_dir=Path("outputs/scene01"),
        ns_train_bin="ns-train-custom",
    )

    command = run_splatfacto_training(spec, dry_run=True)

    assert command[0] == "ns-train-custom"


def test_validate_training_spec_raises_for_missing_data_dir(tmp_path):
    spec = SplatfactoTrainingSpec(
        data_path=tmp_path / "missing_scene",
        output_dir=tmp_path / "outputs",
    )

    try:
        validate_training_spec(spec)
    except FileNotFoundError as exc:
        assert "NerfStudio data directory does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing data directory")
