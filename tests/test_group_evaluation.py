from pathlib import Path
import sys

import json
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gsegmenter.mapping.evaluation import (  # noqa: E402
    classify_gaussian_grouping,
    evaluate_gaussian_grouping,
    summarize_gaussian_comparison,
)


def test_evaluate_gaussian_grouping_matches_best_iou_pairs(tmp_path: Path) -> None:
    gt_object_ids = np.asarray([10, 10, 10, 11, 11, -1], dtype=np.int32)
    pred_object_ids = np.asarray([20, 20, -1, 21, 20, 22], dtype=np.int32)

    gt_groups_json = tmp_path / "gt_groups.json"
    gt_groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {"object_id": 10, "label": "chair", "gaussian_count": 3},
                    {"object_id": 11, "label": "table", "gaussian_count": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_gaussian_grouping(
        gt_object_ids,
        pred_object_ids,
        min_iou=0.1,
        gt_groups_json=gt_groups_json,
    )

    assert metrics.matched_pair_count == 2
    assert metrics.gt_group_count == 2
    assert metrics.pred_group_count == 3
    assert metrics.gt_unmatched_group_count == 0
    assert metrics.pred_unmatched_group_count == 1
    assert np.isclose(metrics.mean_matched_iou, (2 / 4 + 1 / 2) / 2)
    assert any(metric.label == "chair" and metric.matched_group_count == 1 for metric in metrics.label_metrics)
    assert any(metric.label == "table" and metric.matched_group_count == 1 for metric in metrics.label_metrics)


def test_evaluate_gaussian_grouping_handles_empty_predictions() -> None:
    gt_object_ids = np.asarray([10, 10, -1], dtype=np.int32)
    pred_object_ids = np.asarray([-1, -1, -1], dtype=np.int32)

    metrics = evaluate_gaussian_grouping(gt_object_ids, pred_object_ids, min_iou=0.1)

    assert metrics.matched_pair_count == 0
    assert metrics.pred_group_count == 0
    assert metrics.gt_group_count == 1
    assert metrics.gt_group_recall == 0.0
    assert metrics.pred_group_precision == 0.0


def test_classify_gaussian_grouping_separates_match_miss_fp_and_conflict() -> None:
    gt_object_ids = np.asarray([10, 10, 11, -1, 12, -1], dtype=np.int32)
    pred_object_ids = np.asarray([20, -1, 21, 22, 20, -1], dtype=np.int32)

    metrics = evaluate_gaussian_grouping(gt_object_ids, pred_object_ids, min_iou=0.1)
    categories = classify_gaussian_grouping(gt_object_ids, pred_object_ids, metrics.matches)
    summary = summarize_gaussian_comparison(categories)

    assert categories.tolist() == [4, 2, 1, 3, 1, 0]
    assert summary["matched"] == 2
    assert summary["gt_only"] == 1
    assert summary["pred_only"] == 1
    assert summary["conflict"] == 1
    assert summary["background"] == 1
