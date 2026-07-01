from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.data.video_preparation import (  # noqa: E402
    VideoPreparationSpec,
    build_ns_process_video_command,
    validate_video_preparation_spec,
)


def test_build_ns_process_video_command_includes_core_flags() -> None:
    spec = VideoPreparationSpec(
        video_path=Path(r"C:\captures\room.mp4"),
        output_dir=Path(r"C:\data\room_scene"),
        num_frames_target=240,
        camera_type="perspective",
        matching_method="sequential",
        sfm_tool="colmap",
        num_downscales=2,
        crop_bottom=0.1,
        colmap_cmd="colmap",
        gpu=False,
        verbose=True,
        ns_process_data_bin=r"C:\env\Scripts\ns-process-data.exe",
        extra_args=("--use-sfm-depth", "True"),
    )

    command = build_ns_process_video_command(spec)

    assert command[:2] == [r"C:\env\Scripts\ns-process-data.exe", "video"]
    assert "--data" in command and str(spec.video_path) in command
    assert "--output-dir" in command and str(spec.output_dir) in command
    assert "--num-frames-target" in command and "240" in command
    assert "--crop-bottom" in command and "0.1" in command
    assert "--no-gpu" in command
    assert "--verbose" in command
    assert command[-2:] == ["--use-sfm-depth", "True"]


def test_validate_video_preparation_spec_rejects_invalid_crop(tmp_path: Path) -> None:
    video_path = tmp_path / "capture.mp4"
    video_path.write_bytes(b"dummy")
    spec = VideoPreparationSpec(
        video_path=video_path,
        output_dir=tmp_path / "scene",
        crop_bottom=1.5,
    )

    with pytest.raises(ValueError, match="crop-bottom"):
        validate_video_preparation_spec(spec)
