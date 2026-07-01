from __future__ import annotations

import re

import numpy as np

from gsegmenter.mapping.category_discovery import (
    CategoryDiscoverySpec,
    accumulate_category_votes,
    build_category_instance_ids,
    match_category,
)
from gsegmenter.mapping.lifting import VoteEvidence


def test_match_category_uses_first_matching_regex() -> None:
    specs = [
        CategoryDiscoverySpec("chair", re.compile("chair", re.IGNORECASE)),
        CategoryDiscoverySpec("table", re.compile("table", re.IGNORECASE)),
    ]

    assert match_category("large chair", specs) == 0
    assert match_category("coffee table", specs) == 1
    assert match_category("window", specs) is None


def test_accumulate_category_votes_builds_vote_and_support_tables() -> None:
    evidences = [
        VoteEvidence(0, 0, np.array([0, 1], dtype=np.int64), np.array([0.5, 1.0], dtype=np.float32)),
        VoteEvidence(1, 0, np.array([1, 2], dtype=np.int64), np.array([0.25, 0.75], dtype=np.float32)),
    ]

    votes, support = accumulate_category_votes(
        evidences,
        [0, 1],
        gaussian_count=3,
        category_count=2,
    )

    np.testing.assert_allclose(votes[:, 0], [0.5, 1.0, 0.0])
    np.testing.assert_allclose(votes[:, 1], [0.0, 0.25, 0.75])
    assert support[:, 0].tolist() == [1, 1, 0]
    assert support[:, 1].tolist() == [0, 1, 1]


def test_build_category_instance_ids_clusters_best_category_votes() -> None:
    xyz = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [1.04, 0.00, 0.00],
            [5.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    votes = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    support = np.ones_like(votes, dtype=np.int32)

    instance_ids, proposals = build_category_instance_ids(
        xyz=xyz,
        category_votes=votes,
        support_counts=support,
        category_names=["chair", "table"],
        min_vote_weight=0.5,
        min_support_count=1,
        voxel_size=0.05,
        min_voxel_count=1,
        min_gaussians=2,
    )

    assert instance_ids.tolist() == [0, 0, 1, 1, -1]
    assert [proposal.category for proposal in proposals] == ["chair", "chair"]


def test_build_category_instance_ids_can_reject_background_dominant_gaussians() -> None:
    xyz = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [1.04, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    votes = np.ones((4, 1), dtype=np.float32)
    support = np.asarray([[4], [4], [1], [1]], dtype=np.int32)
    background_support = np.asarray([1, 1, 9, 9], dtype=np.int32)

    instance_ids, proposals = build_category_instance_ids(
        xyz=xyz,
        category_votes=votes,
        support_counts=support,
        background_support_counts=background_support,
        category_names=["chair"],
        min_vote_weight=0.5,
        min_support_count=1,
        min_foreground_ratio=0.5,
        voxel_size=0.05,
        min_voxel_count=1,
        min_gaussians=2,
    )

    assert instance_ids.tolist() == [0, 0, -1, -1]
    assert len(proposals) == 1
