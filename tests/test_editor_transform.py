from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.editor.transform import apply_object_transform, apply_object_transform_about_pivot  # noqa: E402


def test_apply_object_transform_preserves_unselected_rows() -> None:
    means = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rotations = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    object_ids = torch.tensor([7, 8], dtype=torch.int64)
    rotation_matrix = torch.eye(3, dtype=torch.float32)
    translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    new_means, new_rotations = apply_object_transform(
        means, rotations, object_ids, 7, translation, rotation_matrix
    )

    assert torch.allclose(new_means[0], torch.tensor([2.0, 2.0, 3.0]))
    assert torch.allclose(new_means[1], means[1])
    assert torch.allclose(new_rotations[1], rotations[1])


def test_apply_object_transform_about_pivot_rotates_around_object_centroid() -> None:
    means = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rotations = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    object_ids = torch.tensor([5, 5, 9], dtype=torch.int64)
    rotation_matrix = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    new_means, _ = apply_object_transform_about_pivot(
        means,
        rotations,
        object_ids,
        5,
        translation=torch.zeros(3, dtype=torch.float32),
        rotation_matrix=rotation_matrix,
    )

    expected = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [2.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(new_means, expected, atol=1e-5)
