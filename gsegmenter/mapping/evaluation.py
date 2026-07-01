"""Evaluation helpers for comparing predicted Gaussian groups to ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


GAUSSIAN_COMPARISON_LABELS = {
    0: "background",
    1: "matched",
    2: "gt_only",
    3: "pred_only",
    4: "conflict",
}


@dataclass(slots=True)
class GroupMatch:
    """One-to-one match between a GT group and a predicted group."""

    gt_object_id: int
    pred_object_id: int
    intersection_count: int
    union_count: int
    iou: float
    gt_gaussian_count: int
    pred_gaussian_count: int
    gt_label: str | None = None


@dataclass(slots=True)
class LabelMetric:
    """Per-label matching summary derived from matched group pairs."""

    label: str
    gt_group_count: int
    matched_group_count: int
    mean_matched_iou: float


@dataclass(slots=True)
class GaussianGroupingMetrics:
    """Scene-level summary for predicted-vs-ground-truth Gaussian grouping."""

    total_gaussians: int
    gt_group_count: int
    pred_group_count: int
    gt_assigned_gaussians: int
    pred_assigned_gaussians: int
    co_assigned_gaussians: int
    matched_pair_count: int
    gt_unmatched_group_count: int
    pred_unmatched_group_count: int
    mean_matched_iou: float
    matched_gt_gaussians: int
    matched_pred_gaussians: int
    gt_group_recall: float
    pred_group_precision: float
    gt_gaussian_recall: float
    pred_gaussian_precision: float
    matches: tuple[GroupMatch, ...]
    label_metrics: tuple[LabelMetric, ...] = ()


def _load_group_labels(groups_json: Path | None) -> dict[int, str]:
    if groups_json is None:
        return {}
    payload = json.loads(Path(groups_json).read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for group in payload.get("groups", []):
        object_id = int(group.get("object_id", group.get("global_object_id", -1)))
        label = group.get("label")
        if object_id >= 0 and isinstance(label, str) and label:
            labels[object_id] = label
    return labels


def _positive_id_counts(object_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = object_ids[object_ids >= 0].astype(np.int64)
    if positive.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    ids, counts = np.unique(positive, return_counts=True)
    return ids.astype(np.int64), counts.astype(np.int64)


def _build_contingency(gt_object_ids: np.ndarray, pred_object_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.logical_and(gt_object_ids >= 0, pred_object_ids >= 0)
    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    pairs = np.stack([gt_object_ids[valid], pred_object_ids[valid]], axis=1).astype(np.int64)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    return unique_pairs[:, 0], unique_pairs[:, 1], counts.astype(np.int64)


def evaluate_gaussian_grouping(
    gt_object_ids: np.ndarray,
    pred_object_ids: np.ndarray,
    *,
    min_iou: float = 0.1,
    gt_groups_json: Path | None = None,
) -> GaussianGroupingMetrics:
    """Compare predicted Gaussian object ids against a GT Gaussian grouping.

    Args:
        gt_object_ids: `(N,)` GT assignment array where `-1` marks unknown.
        pred_object_ids: `(N,)` predicted assignment array where `-1` marks unknown.
        min_iou: Minimum IoU required for a one-to-one group match.
        gt_groups_json: Optional GT group summary with labels for per-label metrics.
    """

    gt_object_ids = np.asarray(gt_object_ids, dtype=np.int64)
    pred_object_ids = np.asarray(pred_object_ids, dtype=np.int64)
    if gt_object_ids.shape != pred_object_ids.shape:
        raise ValueError(
            f"GT/pred object id arrays must share the same shape, got {gt_object_ids.shape} and {pred_object_ids.shape}"
        )

    gt_ids, gt_counts = _positive_id_counts(gt_object_ids)
    pred_ids, pred_counts = _positive_id_counts(pred_object_ids)
    gt_count_lookup = {int(object_id): int(count) for object_id, count in zip(gt_ids, gt_counts)}
    pred_count_lookup = {int(object_id): int(count) for object_id, count in zip(pred_ids, pred_counts)}
    gt_label_lookup = _load_group_labels(gt_groups_json)

    pair_gt_ids, pair_pred_ids, pair_intersections = _build_contingency(gt_object_ids, pred_object_ids)
    candidate_matches: list[GroupMatch] = []
    for gt_object_id, pred_object_id, intersection_count in zip(pair_gt_ids, pair_pred_ids, pair_intersections):
        gt_gaussian_count = gt_count_lookup[int(gt_object_id)]
        pred_gaussian_count = pred_count_lookup[int(pred_object_id)]
        union_count = gt_gaussian_count + pred_gaussian_count - int(intersection_count)
        iou = 0.0 if union_count <= 0 else float(intersection_count) / float(union_count)
        if iou < float(min_iou):
            continue
        candidate_matches.append(
            GroupMatch(
                gt_object_id=int(gt_object_id),
                pred_object_id=int(pred_object_id),
                intersection_count=int(intersection_count),
                union_count=int(union_count),
                iou=iou,
                gt_gaussian_count=int(gt_gaussian_count),
                pred_gaussian_count=int(pred_gaussian_count),
                gt_label=gt_label_lookup.get(int(gt_object_id)),
            )
        )

    candidate_matches.sort(
        key=lambda match: (match.iou, match.intersection_count, -match.union_count),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[GroupMatch] = []
    for match in candidate_matches:
        if match.gt_object_id in used_gt or match.pred_object_id in used_pred:
            continue
        used_gt.add(match.gt_object_id)
        used_pred.add(match.pred_object_id)
        matches.append(match)

    matched_gt_gaussians = int(sum(match.gt_gaussian_count for match in matches))
    matched_pred_gaussians = int(sum(match.pred_gaussian_count for match in matches))
    gt_assigned_gaussians = int(np.count_nonzero(gt_object_ids >= 0))
    pred_assigned_gaussians = int(np.count_nonzero(pred_object_ids >= 0))
    co_assigned_gaussians = int(np.count_nonzero(np.logical_and(gt_object_ids >= 0, pred_object_ids >= 0)))

    mean_matched_iou = 0.0 if not matches else float(np.mean([match.iou for match in matches]))
    gt_group_recall = 0.0 if len(gt_ids) == 0 else len(matches) / float(len(gt_ids))
    pred_group_precision = 0.0 if len(pred_ids) == 0 else len(matches) / float(len(pred_ids))
    gt_gaussian_recall = 0.0 if gt_assigned_gaussians == 0 else matched_gt_gaussians / float(gt_assigned_gaussians)
    pred_gaussian_precision = 0.0 if pred_assigned_gaussians == 0 else matched_pred_gaussians / float(pred_assigned_gaussians)

    label_metrics: list[LabelMetric] = []
    if gt_label_lookup:
        labels = sorted({label for label in gt_label_lookup.values()})
        for label in labels:
            label_gt_ids = [object_id for object_id, object_label in gt_label_lookup.items() if object_label == label]
            label_matches = [match for match in matches if match.gt_label == label]
            label_metrics.append(
                LabelMetric(
                    label=label,
                    gt_group_count=len(label_gt_ids),
                    matched_group_count=len(label_matches),
                    mean_matched_iou=0.0 if not label_matches else float(np.mean([match.iou for match in label_matches])),
                )
            )

    return GaussianGroupingMetrics(
        total_gaussians=int(gt_object_ids.shape[0]),
        gt_group_count=int(len(gt_ids)),
        pred_group_count=int(len(pred_ids)),
        gt_assigned_gaussians=gt_assigned_gaussians,
        pred_assigned_gaussians=pred_assigned_gaussians,
        co_assigned_gaussians=co_assigned_gaussians,
        matched_pair_count=int(len(matches)),
        gt_unmatched_group_count=int(len(gt_ids) - len(matches)),
        pred_unmatched_group_count=int(len(pred_ids) - len(matches)),
        mean_matched_iou=mean_matched_iou,
        matched_gt_gaussians=matched_gt_gaussians,
        matched_pred_gaussians=matched_pred_gaussians,
        gt_group_recall=gt_group_recall,
        pred_group_precision=pred_group_precision,
        gt_gaussian_recall=gt_gaussian_recall,
        pred_gaussian_precision=pred_gaussian_precision,
        matches=tuple(matches),
        label_metrics=tuple(label_metrics),
    )


def metrics_to_json_dict(metrics: GaussianGroupingMetrics) -> dict:
    """Convert metrics dataclasses into a JSON-serializable dictionary."""

    payload = asdict(metrics)
    return payload


def classify_gaussian_grouping(
    gt_object_ids: np.ndarray,
    pred_object_ids: np.ndarray,
    matches: tuple[GroupMatch, ...] | list[GroupMatch],
) -> np.ndarray:
    """Classify each Gaussian by GT/pred agreement status.

    Returns an `(N,)` int8 array with the following meaning:
    - `0`: background / neither GT nor prediction assigned
    - `1`: matched pair agreement
    - `2`: GT assigned but prediction missing
    - `3`: prediction assigned but GT missing
    - `4`: both assigned, but the predicted object does not match the GT object
    """

    gt_object_ids = np.asarray(gt_object_ids, dtype=np.int64)
    pred_object_ids = np.asarray(pred_object_ids, dtype=np.int64)
    if gt_object_ids.shape != pred_object_ids.shape:
        raise ValueError(
            f"GT/pred object id arrays must share the same shape, got {gt_object_ids.shape} and {pred_object_ids.shape}"
        )

    categories = np.zeros(gt_object_ids.shape, dtype=np.int8)
    gt_assigned = gt_object_ids >= 0
    pred_assigned = pred_object_ids >= 0

    categories[np.logical_and(gt_assigned, ~pred_assigned)] = np.int8(2)
    categories[np.logical_and(~gt_assigned, pred_assigned)] = np.int8(3)

    matched_pair_lookup = {
        (int(match.gt_object_id), int(match.pred_object_id))
        for match in matches
    }
    both_assigned = np.logical_and(gt_assigned, pred_assigned)
    if np.any(both_assigned):
        paired_gt = gt_object_ids[both_assigned].astype(np.int64)
        paired_pred = pred_object_ids[both_assigned].astype(np.int64)
        paired_match = np.fromiter(
            ((int(gt_id), int(pred_id)) in matched_pair_lookup for gt_id, pred_id in zip(paired_gt, paired_pred)),
            count=paired_gt.shape[0],
            dtype=bool,
        )
        categories[both_assigned] = np.where(paired_match, np.int8(1), np.int8(4))
    return categories


def summarize_gaussian_comparison(categories: np.ndarray) -> dict[str, int]:
    """Return counts for the comparison labels in ``GAUSSIAN_COMPARISON_LABELS``."""

    categories = np.asarray(categories, dtype=np.int8)
    return {
        label: int(np.count_nonzero(categories == code))
        for code, label in GAUSSIAN_COMPARISON_LABELS.items()
    }
