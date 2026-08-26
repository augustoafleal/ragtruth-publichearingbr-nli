from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


CONDITIONS = ("en_attention", "en_set", "pt_attention", "pt_set")
METRICS = ("auprc", "auroc", "brier")
CONTRASTS = {
    "en_set_minus_en_attention": ("en_set", "en_attention"),
    "pt_set_minus_pt_attention": ("pt_set", "pt_attention"),
    "pt_attention_minus_en_attention": ("pt_attention", "en_attention"),
    "pt_set_minus_en_set": ("pt_set", "en_set"),
}


def _metric(metric: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if metric == "auprc":
        return float(average_precision_score(labels, scores))
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    if metric == "brier":
        return float(brier_score_loss(labels, scores))
    raise ValueError(f"Unknown metric: {metric}")


def _weighted_average_precision(
    labels_sorted: np.ndarray,
    score_group_starts: np.ndarray,
    row_weights_sorted: np.ndarray,
) -> float:
    weighted_labels = labels_sorted * row_weights_sorted
    threshold_true = np.add.reduceat(weighted_labels, score_group_starts)
    threshold_rows = np.add.reduceat(row_weights_sorted, score_group_starts)
    active = threshold_rows > 0
    threshold_true = threshold_true[active]
    threshold_rows = threshold_rows[active]
    total_positive = float(threshold_true.sum())
    if total_positive == 0:
        return float("nan")
    cumulative_true = np.cumsum(threshold_true)
    cumulative_rows = np.cumsum(threshold_rows)
    precision = cumulative_true / cumulative_rows
    return float(np.sum(precision * threshold_true) / total_positive)


def validate_factorial_frames(
    frames: dict[str, dict[int, pd.DataFrame]],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    expected_rows: int = 4235,
    expected_positives: int = 501,
    expected_hearings: int = 206,
    score_column: str = "probability",
) -> dict[str, Any]:
    if set(frames) != set(CONDITIONS):
        raise ValueError(f"Expected conditions {CONDITIONS}, got {tuple(sorted(frames))}")
    reference: pd.DataFrame | None = None
    audit: dict[str, Any] = {}
    for condition in CONDITIONS:
        if set(frames[condition]) != set(seeds):
            raise ValueError(f"Missing seeds for {condition}: {sorted(frames[condition])}")
        audit[condition] = {}
        for seed in seeds:
            frame = frames[condition][seed]
            required = {"example_id", "hearing_id", "label", score_column}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing columns for {condition}/seed_{seed}: {sorted(required - set(frame.columns))}")
            current = frame[["example_id", "hearing_id", "label", score_column]].copy()
            current["example_id"] = current["example_id"].astype(str)
            current["hearing_id"] = current["hearing_id"].astype(str)
            if len(current) != expected_rows or not current.example_id.is_unique or current.hearing_id.nunique() != expected_hearings:
                raise ValueError(f"Invalid population for {condition}/seed_{seed}")
            if current.isna().any().any() or set(current.label.astype(int).unique()) != {0, 1} or int(current.label.astype(int).sum()) != expected_positives:
                raise ValueError(f"Invalid labels for {condition}/seed_{seed}")
            scores = current[score_column].to_numpy(float)
            if not np.isfinite(scores).all() or not ((scores >= 0).all() and (scores <= 1).all()):
                raise ValueError(f"Invalid probabilities for {condition}/seed_{seed}")
            if reference is None:
                reference = current[["example_id", "hearing_id", "label"]].set_index("example_id").sort_index()
            else:
                indexed = current.set_index("example_id").sort_index()
                if not reference.index.equals(indexed.index):
                    raise ValueError(f"example_id mismatch for {condition}/seed_{seed}")
                if not reference["hearing_id"].equals(indexed["hearing_id"]):
                    raise ValueError(f"hearing_id mismatch for {condition}/seed_{seed}")
                if not reference["label"].astype(int).equals(indexed["label"].astype(int)):
                    raise ValueError(f"label mismatch for {condition}/seed_{seed}")
            audit[condition][str(seed)] = {"examples": len(current), "hearings": int(current.hearing_id.nunique()), "positives": int(current.label.sum())}
    assert reference is not None
    return {"examples": expected_rows, "positives": expected_positives, "prevalence": expected_positives / expected_rows,
            "hearings": expected_hearings, "exact_example_alignment": True, "exact_hearing_alignment": True,
            "exact_label_alignment": True, "probabilities_valid": True, "per_condition": audit}


def _summary(values: np.ndarray, confidence_level: float) -> dict[str, float | int]:
    alpha = 1.0 - confidence_level
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci_lower": float(np.quantile(values, alpha / 2)), "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
            "bootstrap_support_probability_delta_gt_zero": float(np.mean(values > 0)), "n_valid": int(len(values))}


def run_factorial_bootstrap(
    frames: dict[str, dict[int, pd.DataFrame]],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_replicates: int = 10000,
    seed: int = 20260815,
    confidence_level: float = 0.95,
    score_column: str = "probability",
    metrics: tuple[str, ...] = ("auprc",),
    expected_rows: int = 4235,
    expected_positives: int = 501,
    expected_hearings: int = 206,
) -> dict[str, Any]:
    if not metrics or not set(metrics).issubset(METRICS):
        raise ValueError(f"metrics must be a non-empty subset of {METRICS}, got {metrics}")
    population = validate_factorial_frames(
        frames,
        seeds=seeds,
        expected_rows=expected_rows,
        expected_positives=expected_positives,
        expected_hearings=expected_hearings,
        score_column=score_column,
    )
    reference = frames["en_attention"][seeds[0]].copy()
    reference["example_id"] = reference["example_id"].astype(str)
    reference = reference.set_index("example_id").sort_index()
    labels = reference["label"].to_numpy(int)
    groups = sorted(reference["hearing_id"].astype(str).unique())
    group_values = reference["hearing_id"].astype(str).to_numpy()
    group_indices = [np.flatnonzero(group_values == group) for group in groups]
    group_codes = np.searchsorted(np.asarray(groups), group_values)

    scores: dict[str, dict[int, np.ndarray]] = {}
    ap_specs: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    observed: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        scores[condition] = {}
        ap_specs[condition] = {}
        observed[condition] = {}
        for paired_seed in seeds:
            frame = frames[condition][paired_seed].copy()
            frame["example_id"] = frame["example_id"].astype(str)
            indexed = frame.set_index("example_id").sort_index()
            values = indexed.loc[reference.index, score_column].to_numpy(float)
            scores[condition][paired_seed] = values
            order = np.argsort(-values, kind="mergesort")
            sorted_scores = values[order]
            starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
            ap_specs[condition][paired_seed] = (labels[order], order, starts)
            observed[condition][str(paired_seed)] = {metric: _metric(metric, labels, values) for metric in METRICS}

    observed_campaign: dict[str, dict[str, float]] = {}
    for contrast, (left, right) in CONTRASTS.items():
        observed_campaign[contrast] = {metric: float(np.mean([observed[left][str(paired_seed)][metric] - observed[right][str(paired_seed)][metric] for paired_seed in seeds])) for metric in METRICS}
    observed_campaign["interaction"] = {metric: observed_campaign["pt_set_minus_pt_attention"][metric] - observed_campaign["en_set_minus_en_attention"][metric] for metric in METRICS}

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate_id in range(n_replicates):
        sampled_positions = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([group_indices[position] for position in sampled_positions])
        hearing_multiplicity = np.bincount(sampled_positions, minlength=len(groups))
        row_weights = hearing_multiplicity[group_codes]
        replicate_values: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            replicate_values[condition] = {}
            for metric in metrics:
                if metric == "auprc":
                    replicate_values[condition][metric] = float(np.mean([
                        _weighted_average_precision(
                            labels_sorted,
                            starts,
                            row_weights[order],
                        )
                        for labels_sorted, order, starts in (ap_specs[condition][paired_seed] for paired_seed in seeds)
                    ]))
                else:
                    replicate_values[condition][metric] = float(np.mean([_metric(metric, labels[indices], scores[condition][paired_seed][indices]) for paired_seed in seeds]))
        for contrast, (left, right) in CONTRASTS.items():
            for metric in metrics:
                rows.append({"replicate_id": replicate_id, "contrast": contrast, "metric": metric, "delta": replicate_values[left][metric] - replicate_values[right][metric]})
        for metric in metrics:
            rows.append({"replicate_id": replicate_id, "contrast": "interaction", "metric": metric,
                         "delta": (replicate_values["pt_set"][metric] - replicate_values["pt_attention"][metric]) - (replicate_values["en_set"][metric] - replicate_values["en_attention"][metric])})

    replicate_frame = pd.DataFrame(rows)
    summaries = {contrast: {metric: _summary(replicate_frame.loc[(replicate_frame.contrast == contrast) & (replicate_frame.metric == metric), "delta"].to_numpy(float), confidence_level) for metric in metrics} for contrast in (*CONTRASTS, "interaction")}
    return {"population": population, "observed": observed, "observed_campaign": observed_campaign, "summaries": summaries,
            "replicates": replicate_frame, "protocol": {"paired": True, "group_key": "hearing_id", "n_replicates": n_replicates,
                                                          "seed": seed, "confidence_level": confidence_level, "same_samples_all_conditions": True,
                                                          "same_samples_all_seeds": True, "campaign_statistic": "mean paired seed effects",
                                                          "metrics": list(metrics)}}
