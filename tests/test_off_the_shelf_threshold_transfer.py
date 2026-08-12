from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.metrics import binary_metrics, select_threshold
from ragtruth_transfer.off_the_shelf_threshold_transfer import (
    OPERATOR,
    ThresholdTransferConfig,
    _comparison,
    _operating_metrics,
    _select_baseline_thresholds,
    _signature,
    _validation_predictions,
)
from ragtruth_transfer.publichearing_off_the_shelf import _normalize_labels


def _config() -> ThresholdTransferConfig:
    return ThresholdTransferConfig.from_yaml(Path("configs/ragtruth_off_the_shelf_threshold_transfer.yaml"))


def test_uses_the_frozen_off_the_shelf_model_revision_and_no_lora():
    config = _config()
    assert config.model_id == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    assert config.model_revision == "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    assert config.to_dict()["model"]["loader"] == "AutoModelForSequenceClassification"
    assert config.to_dict()["model"]["lora_used"] is False
    assert config.truncation == "only_first" and OPERATOR == ">="


def test_dynamic_nli_mapping_is_checked_from_model_configuration():
    class Config:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

    class Model:
        config = Config()

    assert _normalize_labels(Model()) == {"entailment": 0, "neutral": 1, "contradiction": 2}

    class ReorderedConfig:
        id2label = {0: "neutral", 1: "contradiction", 2: "entailment"}
        label2id = {"neutral": 0, "contradiction": 1, "entailment": 2}

    class ReorderedModel:
        config = ReorderedConfig()

    assert _normalize_labels(ReorderedModel()) == {"neutral": 0, "contradiction": 1, "entailment": 2}

    class GenericConfig:
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

    class GenericModel:
        config = GenericConfig()

    with pytest.raises(ValueError):
        _normalize_labels(GenericModel())


def test_masked_slot_cannot_win_max_entailment_or_change_score():
    config = _config()
    validation = [{"example_id": "x", "source_id": "s", "label": 1, "claim": "claim", "evidence": ["valid", "valid two", "valid three", ""], "evidence_mask": [True, True, True, False]}]
    entailment = np.array([[0.30, 0.60, 0.40, np.nan]])
    neutral = np.array([[0.50, 0.20, 0.30, np.nan]])
    contradiction = np.array([[0.20, 0.20, 0.30, np.nan]])
    frame = _validation_predictions(validation, entailment, neutral, contradiction, config, "signature")
    assert frame.loc[0, "max_entailment"] == 0.60
    assert frame.loc[0, "max_entailment_chunk_index"] == 2
    assert frame.loc[0, "hallucination_score"] == 0.40
    assert np.isnan(frame.loc[0, "entailment_probability_chunk_4"])


def test_off_the_shelf_thresholds_reuse_exact_confirmatory_selector():
    config = _config()
    frame = pd.DataFrame({"label": [0, 0, 1, 1, 1], "hallucination_score": [0.10, 0.45, 0.45, 0.80, 0.90]})
    thresholds, metrics = _select_baseline_thresholds(frame, config)
    for regime, criterion in (("best_f1", "f1"), ("fpr10", "fpr10")):
        threshold, expected, feasible = select_threshold(frame.label.to_numpy(bool), frame.hallucination_score.to_numpy(float), criterion, max_fpr=0.10)
        assert thresholds[regime]["threshold"] == threshold
        assert thresholds[regime]["development_metrics"] == expected
        assert thresholds[regime]["source_metrics"] == expected
        assert thresholds[regime]["selection_dataset_signature"] == config.dataset_signature
        assert thresholds[regime]["selection_split_signature"] == config.split_signature
        assert thresholds[regime]["constraint_feasible"] is feasible
        assert metrics[regime]["decision_operator"] == ">="
    assert thresholds["fpr10"]["achieved_source_fpr"] == thresholds["fpr10"]["source_metrics"]["FPR"]


def test_operating_metrics_and_lora_deltas_have_intended_orientation():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    metric = _operating_metrics(labels, np.array([0.1, 0.9, 0.8, 0.7]), 0.5, "RAGTruth validation")
    assert metric["TP"] == 2 and metric["FP"] == 1 and metric["TN"] == 1 and metric["FN"] == 0
    assert metric["secondary_metric_due_to_class_imbalance"] is True
    baseline = {"best_f1": metric, "fpr10": metric}
    lora_audit = {"seeds": {"0": {"regimes": {regime: {"development_metrics": metric} for regime in ("best_f1", "fpr10")}}}}
    better = dict(metric); better.update({"F1": 0.9, "Recall": 0.9, "Precision": 0.9, "FPR": 0.1, "MCC": 0.8, "BalancedAccuracy": 0.9})
    target = {"off_the_shelf": baseline, "lora": {"0": {"best_f1": better, "fpr10": better}}}
    comparison = _comparison(baseline, lora_audit, target)
    delta = comparison["lora_minus_baseline"]["0"]["best_f1"]
    assert delta["delta_f1"] > 0 and delta["fpr_reduction"] > 0
    assert comparison["ensemble_used"] is False and comparison["best_seed_selected"] is False
    assert comparison["threshold_selection_uses_publichearing_labels"] is False
    assert comparison["operating_point_transfer"]["off_the_shelf"]["best_f1"]["fpr"]["delta"] == 0.0


def test_score_equal_to_threshold_is_positive_under_shared_operator():
    metric = _operating_metrics(np.array([0, 1], dtype=bool), np.array([0.49, 0.50]), 0.50, "RAGTruth validation")
    assert metric["TP"] == 1 and metric["FN"] == 0 and metric["FP"] == 0


def test_signature_is_independent_of_paths_and_batch_size():
    config = _config()
    validation = {"validation_example_ids_hash": "ids", "validation_evidence_masks_hash": "masks"}
    lora, target = {"lora": "hash"}, {"target": "hash"}
    signature = _signature(config, validation, lora, target)[0]
    moved = replace(config, output_root=Path("/tmp/out"), baseline_run=Path("/tmp/base"), confirmatory_run=Path("/tmp/confirm"), confirmatory_config_path=Path("/tmp/config"), batch_size=1)
    assert signature == _signature(moved, validation, lora, target)[0]
    assert signature != _signature(replace(config, model_revision="different"), validation, lora, target)[0]
    assert signature != _signature(replace(config, target_fpr=0.05), validation, lora, target)[0]
    assert signature != _signature(config, {"validation_example_ids_hash": "different", "validation_evidence_masks_hash": "masks"}, lora, target)[0]
    assert signature != _signature(config, validation, lora, {"target": "different"})[0]
    assert _signature(config, validation, lora, target)[1]["decision_operator"] == ">="


def test_binary_metrics_are_the_shared_confusion_matrix_definition():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    prediction = np.array([0, 1, 1, 1], dtype=bool)
    metric = binary_metrics(labels, prediction)
    assert (metric["TN"], metric["FP"], metric["FN"], metric["TP"]) == (1, 1, 0, 2)
    assert np.isclose(metric["FPR"], 0.5) and np.isclose(metric["Specificity"], 0.5)
