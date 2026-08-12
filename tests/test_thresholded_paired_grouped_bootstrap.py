from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.thresholded_paired_grouped_bootstrap import (
    METRICS,
    OPERATOR,
    ThresholdedBootstrapConfig,
    _bootstrap_metric_values,
    _metrics,
    _signature,
    _strict_join,
    _summary,
    _validate_prediction,
)


def _frame(score_name: str, *, ids=("a", "b", "c", "d"), labels=(0, 1, 0, 1), hearings=("h1", "h1", "h2", "h2")) -> pd.DataFrame:
    return pd.DataFrame({"example_id": ids, "hearing_id": hearings, "label": labels, score_name: (.1, .9, .2, .8)})


def test_config_freezes_threshold_and_paired_grouped_contract():
    config = ThresholdedBootstrapConfig.from_yaml(Path("configs/publichearing_thresholded_paired_bootstrap.yaml"))
    assert config.reestimate_thresholds is False
    assert config.regimes == ("best_f1", "fpr10") and config.seeds == (0, 1, 2)
    assert config.same_samples_across_methods and config.same_samples_across_seeds and config.same_samples_across_threshold_regimes
    assert OPERATOR == ">=" and set(config.primary_metrics + config.secondary_metrics) == set(METRICS)


def test_strict_pairing_rejects_id_label_and_hearing_mismatches():
    base = pd.DataFrame({"example_id": [f"e{i}" for i in range(4235)], "hearing_id": [f"h{i % 206}" for i in range(4235)], "label": [1] * 501 + [0] * 3734, "hallucination_score": np.linspace(.01, .99, 4235)})
    lora = base.rename(columns={"hallucination_score": "probability"})
    joined, audit = _strict_join(base, {0: lora, 1: lora.copy(), 2: lora.copy()})
    assert len(joined) == 4235 and audit["0"]["pairable_examples"] == 4235
    bad = lora.copy(); bad.loc[0, "example_id"] = "missing"
    with pytest.raises(ValueError): _strict_join(base, {0: bad, 1: lora, 2: lora})
    bad = lora.copy(); bad.loc[0, "label"] = 0
    with pytest.raises(ValueError): _strict_join(base, {0: bad, 1: lora, 2: lora})
    bad = lora.copy(); bad.loc[0, "hearing_id"] = "elsewhere"
    with pytest.raises(ValueError): _strict_join(base, {0: bad, 1: lora, 2: lora})


def test_decision_operator_includes_score_equal_to_threshold_and_metrics_are_correct():
    labels = np.array([0, 0, 1, 1])
    result = _metrics(labels, np.array([.49, .50, .50, .99]), .50)
    assert (result["TN"], result["FP"], result["FN"], result["TP"]) == (1, 1, 0, 2)
    assert result["F1"] == 0.8 and result["Recall"] == 1.0 and result["Precision"] == 2 / 3
    assert result["FPR"] == .5 and result["Specificity"] == .5 and result["BalancedAccuracy"] == .75
    direct = _bootstrap_metric_values(labels, np.array([False, True, True, True]))
    assert np.isclose(direct["mcc"], result["MCC"])
    for metric, key in (("f1", "F1"), ("recall", "Recall"), ("precision", "Precision"), ("fpr", "FPR"), ("specificity", "Specificity"), ("balanced_accuracy", "BalancedAccuracy"), ("accuracy", "Accuracy")):
        assert np.isclose(direct[metric], result[key])


def test_group_sampling_preserves_full_group_and_multiplicity():
    groups = np.array(["A", "A", "B", "C"])
    group_indices = [np.flatnonzero(groups == group) for group in sorted(np.unique(groups))]
    sampled_positions = np.array([0, 0, 2])
    sampled = np.concatenate([group_indices[position] for position in sampled_positions])
    assert sampled.tolist() == [0, 1, 0, 1, 3]


def test_invalid_scores_and_duplicate_ids_are_rejected(tmp_path):
    complete = pd.DataFrame({"example_id": [f"e{i}" for i in range(4235)], "hearing_id": [f"h{i % 206}" for i in range(4235)], "label": [1] * 501 + [0] * 3734, "hallucination_score": np.linspace(.01, .99, 4235)})
    path = tmp_path / "predictions.parquet"; complete.to_parquet(path, index=False)
    assert len(_validate_prediction(path, "hallucination_score")) == 4235
    bad = complete.copy(); bad.loc[0, "hallucination_score"] = np.nan; bad.to_parquet(path, index=False)
    with pytest.raises(ValueError): _validate_prediction(path, "hallucination_score")
    bad = complete.copy(); bad.loc[1, "example_id"] = bad.loc[0, "example_id"]; bad.to_parquet(path, index=False)
    with pytest.raises(ValueError): _validate_prediction(path, "hallucination_score")


def test_percentile_summary_and_support_probability():
    result = _summary(np.array([-.2, .1, .3, .4]), .95)
    assert result["n_valid"] == 4 and result["bootstrap_support_probability"] == .75
    assert result["ci_lower"] < result["ci_upper"]


def test_signature_does_not_depend_on_paths_but_changes_with_thresholds():
    config = ThresholdedBootstrapConfig.from_yaml(Path("configs/publichearing_thresholded_paired_bootstrap.yaml"))
    joined = pd.DataFrame({"example_id": ["e"], "hearing_id": ["h"]})
    hashes = {"baseline_predictions.parquet": "a", "seed_0_predictions.parquet": "b"}
    thresholds = {"baseline": {"best_f1": .8}, "lora": {0: {"best_f1": .5}}}
    signature = _signature(config, hashes, joined, thresholds)[0]
    moved = replace(config, output_root=Path("/tmp/out"), baseline_run=Path("/tmp/base"), confirmatory_run=Path("/tmp/confirm"))
    assert signature == _signature(moved, hashes, joined, thresholds)[0]
    assert signature != _signature(config, hashes, joined, {"baseline": {"best_f1": .7}, "lora": {0: {"best_f1": .5}}})[0]
