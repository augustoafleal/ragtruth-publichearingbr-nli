from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    result: dict[str, float | int] = {
        "N": int(len(y_true)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": safe_divide(tp, tp + fp),
        "Recall": safe_divide(tp, tp + fn),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "FPR": safe_divide(fp, fp + tn),
        "Specificity": safe_divide(tn, tn + fp),
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        result["AUPRC"] = float(average_precision_score(y_true, scores))
        result["AUROC"] = float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else float("nan")
    return result


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.clip(np.asarray(scores, dtype=float), 0.0, 1.0))
    return np.unique(np.concatenate(([0.0], values, [1.0, np.nextafter(1.0, 2.0)])))


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    criterion: str,
    max_fpr: float = 0.10,
) -> tuple[float, dict[str, float | int], bool]:
    labels = np.asarray(y_true, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if len(labels) != len(values) or len(labels) == 0 or not np.isfinite(values).all():
        raise ValueError("labels/scores inválidos para seleção de threshold")
    candidates: list[tuple[float, dict[str, float | int]]] = []
    order = np.argsort(-values, kind="stable")
    descending_negative_scores = -values[order]
    positive_prefix = np.concatenate(([0], np.cumsum(labels[order], dtype=int)))
    total_positive = int(labels.sum())
    total_negative = int(len(labels) - total_positive)
    for threshold in threshold_candidates(values):
        predicted_positive = int(np.searchsorted(descending_negative_scores, -float(threshold), side="right"))
        tp = int(positive_prefix[predicted_positive]); fp = predicted_positive - tp
        fn = total_positive - tp; tn = total_negative - fp
        precision = safe_divide(tp, tp + fp); recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
        fpr = safe_divide(fp, fp + tn)
        candidates.append((float(threshold), {"F1": f1, "Recall": recall, "FPR": fpr}))

    if criterion == "f1":
        candidates.sort(
            key=lambda item: (
                float(item[1]["F1"]),
                float(item[1]["Recall"]),
                -float(item[1]["FPR"]),
                item[0],
            ),
            reverse=True,
        )
        threshold, _ = candidates[0]
        return threshold, binary_metrics(labels, values >= threshold), True

    if criterion == "fpr10":
        feasible = [item for item in candidates if float(item[1]["FPR"]) <= max_fpr + 1e-12]
        if not feasible:
            candidates.sort(key=lambda item: (float(item[1]["FPR"]), -float(item[1]["Recall"]), -item[0]))
            threshold, _ = candidates[0]
            return threshold, binary_metrics(labels, values >= threshold), False
        feasible.sort(
            key=lambda item: (
                float(item[1]["Recall"]),
                float(item[1]["F1"]),
                -float(item[1]["FPR"]),
                -item[0],
            ),
            reverse=True,
        )
        threshold, _ = feasible[0]
        return threshold, binary_metrics(labels, values >= threshold), True

    raise ValueError(f"Critério desconhecido: {criterion}")


def metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_metrics(y_true, y_pred, scores)[metric])


def paired_cluster_bootstrap(
    y_true: np.ndarray,
    groups: np.ndarray,
    first_pred: np.ndarray,
    first_scores: np.ndarray,
    second_pred: np.ndarray,
    second_scores: np.ndarray,
    metrics: tuple[str, ...] = ("F1", "Recall", "FPR", "MCC", "AUPRC"),
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[dict[str, float | str]]:
    y_true = np.asarray(y_true, dtype=bool)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []

    for metric in metrics:
        point = metric_value(metric, y_true, first_pred, first_scores) - metric_value(
            metric, y_true, second_pred, second_scores
        )
        differences: list[float] = []
        for _ in range(n_resamples):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_positions = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
            first_value = metric_value(
                metric,
                y_true[sampled_positions],
                first_pred[sampled_positions],
                first_scores[sampled_positions],
            )
            second_value = metric_value(
                metric,
                y_true[sampled_positions],
                second_pred[sampled_positions],
                second_scores[sampled_positions],
            )
            differences.append(first_value - second_value)
        values = np.asarray(differences, dtype=float)
        rows.append(
            {
                "metric": metric,
                "difference_first_minus_second": float(point),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "probability_first_better": float(np.mean(values < 0 if metric == "FPR" else values > 0)),
            }
        )
    return rows
