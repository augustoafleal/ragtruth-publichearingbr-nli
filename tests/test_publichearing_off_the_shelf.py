from __future__ import annotations

import json
import numpy as np
import pytest
from dataclasses import replace
from pathlib import Path
import pandas as pd

from ragtruth_transfer.publichearing_off_the_shelf import OffTheShelfConfig, _metrics, _normalize_labels, _resume_from_complete_predictions, _signature


def test_metrics_use_hallucination_orientation_and_no_threshold_metrics():
    rows = [{"label": 0}, {"label": 1}, {"label": 1}, {"label": 0}]
    support = np.array([0.9, 0.2, 0.1, 0.8], dtype=float)
    result = _metrics(rows, 1.0 - support, support)
    assert result["AUPRC"] > 0.9
    assert result["AUROC"] == 1.0
    assert "F1" not in result and "MCC" not in result and "accuracy" not in result
    assert result["sanity_checks"]["hallucination_equals_one_minus_support"]


def test_generic_nli_labels_fail():
    class Config:
        id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

    class Model:
        config = Config()

    with pytest.raises(ValueError, match="mapeamento NLI"):
        _normalize_labels(Model())


def test_label_mapping_is_dynamic_and_unambiguous():
    class Config:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}

    class Model:
        config = Config()

    assert _normalize_labels(Model()) == {"contradiction": 0, "entailment": 1, "neutral": 2}


def test_signature_is_path_independent_but_protocol_dependent():
    config = OffTheShelfConfig.from_yaml(Path("configs/publichearing_off_the_shelf_max_entailment.yaml"))
    baseline = _signature(config, "ids")[0]
    assert baseline == _signature(replace(config, dataset_path=Path("/tmp/other.jsonl"), output_root=Path("/tmp/out")), "ids")[0]
    assert baseline != _signature(config, "ids", premise="claim", hypothesis="chunk")[0]
    assert baseline != _signature(replace(config, model_revision="other", tokenizer_revision="other"), "ids")[0]
    assert baseline != _signature(config, "ids", positive_score="max_entailment")[0]
    assert baseline != _signature(config, "other-ids")[0]


def test_resume_derives_artifacts_without_model(tmp_path):
    class Config:
        model_id = "m"
        model_revision = "r"
        dataset_path = Path("/dataset")
        dataset_sha256 = "sha"
        dataset_revision = "rev"

        def to_dict(self):
            return {"model": self.model_id}

    rows = [{"example_id": f"{i}:0:0", "hearing_id": str(i), "label": i % 2} for i in range(4)]
    entailment = np.asarray([[0.1, 0.7, 0.3, 0.2], [0.8, 0.1, 0.2, 0.3], [0.4, 0.2, 0.1, 0.3], [0.2, 0.3, 0.6, 0.1]])
    frame = pd.DataFrame({
        "example_id": [row["example_id"] for row in rows], "hearing_id": [row["hearing_id"] for row in rows], "label": [row["label"] for row in rows],
        **{f"entailment_chunk_{i + 1}": entailment[:, i] for i in range(4)},
        **{f"neutral_chunk_{i + 1}": 0.1 for i in range(4)},
        **{f"contradiction_chunk_{i + 1}": 0.9 - entailment[:, i] for i in range(4)},
        "max_entailment": entailment.max(axis=1), "max_entailment_chunk_index": entailment.argmax(axis=1) + 1,
        "hallucination_score": 1.0 - entailment.max(axis=1), "run_signature": "sig", "model_id": "m", "model_revision": "r",
    })
    frame.to_parquet(tmp_path / "predictions.parquet", index=False)
    result = _resume_from_complete_predictions(Config(), tmp_path, rows, "example-hash", "sig", {"formula": "x"}, {"references": []})
    assert result["status"] == "completed"
    assert (tmp_path / "metrics.json").is_file()
    assert json.loads((tmp_path / "integrity_audit.json").read_text())["inference_reexecuted"] is False
