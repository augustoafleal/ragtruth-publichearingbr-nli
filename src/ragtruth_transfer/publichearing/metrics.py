from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..metrics import binary_metrics, threshold_candidates


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(labels).size != 2:
        raise ValueError("AUPRC/AUROC requerem as duas classes.")
    return {"AUPRC": float(average_precision_score(labels, probabilities)), "AUROC": float(roc_auc_score(labels, probabilities)), "Brier": float(brier_score_loss(labels, probabilities))}


def select_operating_threshold(labels: np.ndarray, probabilities: np.ndarray, criterion: str, fpr_target: float) -> tuple[float, dict[str, float | int]]:
    candidates = [(float(threshold), binary_metrics(labels, np.asarray(probabilities) >= threshold)) for threshold in threshold_candidates(probabilities)]
    if criterion == "max_f1":
        # At equal F1 a larger threshold is deterministic and avoids needless alerts.
        return max(candidates, key=lambda item: (float(item[1]["F1"]), item[0]))
    if criterion == "fpr_operational":
        feasible = [item for item in candidates if float(item[1]["FPR"]) <= fpr_target + 1e-12]
        if not feasible:
            raise RuntimeError("Nenhum threshold satisfaz o alvo de FPR na validação.")
        return max(feasible, key=lambda item: (float(item[1]["Recall"]), -float(item[1]["FPR"]), item[0]))
    raise ValueError(f"Critério inválido: {criterion}")


def all_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    return {**ranking_metrics(labels, probabilities), **binary_metrics(labels, np.asarray(probabilities) >= threshold)}


def grouped_bootstrap(oof: pd.DataFrame, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = oof.hearing_id.astype(str).unique()
    indexed = {group: oof.loc[oof.hearing_id.astype(str).eq(group)].copy() for group in groups}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    criteria = {"max_f1": "prediction_max_f1", "fpr_operational": "prediction_fpr_operational"}
    for repetition in range(repetitions):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([indexed[group] for group in selected], ignore_index=True)
        labels, probabilities = sample.label.to_numpy(dtype=bool), sample.probability.to_numpy(dtype=float)
        if np.unique(labels).size != 2:
            continue
        ranking = ranking_metrics(labels, probabilities)
        for metric, value in ranking.items():
            rows.append({"repetition": repetition, "criterion": "ranking", "metric": metric, "value": value})
        for criterion, column in criteria.items():
            values = binary_metrics(labels, sample[column].to_numpy(dtype=bool))
            for metric in ("Precision", "Recall", "F1", "BalancedAccuracy", "MCC", "FPR"):
                rows.append({"repetition": repetition, "criterion": criterion, "metric": metric, "value": float(values[metric])})
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("Bootstrap agrupado não produziu amostras com duas classes.")
    observed_rows: list[dict[str, float | str]] = []
    labels, probabilities = oof.label.to_numpy(dtype=bool), oof.probability.to_numpy(dtype=float)
    for metric, value in ranking_metrics(labels, probabilities).items():
        observed_rows.append({"criterion": "ranking", "metric": metric, "observed": value})
    for criterion, column in criteria.items():
        for metric in ("Precision", "Recall", "F1", "BalancedAccuracy", "MCC", "FPR"):
            observed_rows.append({"criterion": criterion, "metric": metric, "observed": float(binary_metrics(labels, oof[column].to_numpy(dtype=bool))[metric])})
    observed = pd.DataFrame(observed_rows)
    summary = samples.groupby(["criterion", "metric"], as_index=False).value.agg(bootstrap_mean="mean", ci_2_5=lambda value: float(np.quantile(value, .025)), ci_97_5=lambda value: float(np.quantile(value, .975)), valid_resamples="count").merge(observed, on=["criterion", "metric"], validate="one_to_one")
    return samples, summary.sort_values(["criterion", "metric"]).reset_index(drop=True)
