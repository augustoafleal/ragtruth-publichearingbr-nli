from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from ragtruth_transfer.config import load_config
from ragtruth_transfer.dataset import BagCollator
from ragtruth_transfer.ragtruth_parquet import load_ragtruth_parquet, split_ragtruth_parquet
from ragtruth_transfer.training import validate_training_data


def _parquet_fixture(tmp_path: Path) -> Path:
    rows = []
    for index, source in enumerate(["s1", "s2", "s3", "s4", "s5", "s6"]):
        rows.append({
            "example_id": f"e{index}",
            "source_id": source,
            "split": "train",
            "claim": f"claim {index}",
            "label": int(index % 2 == 0),
            "evidence_mask": [True, True, False, False],
            "chunk_1": "evidence one",
            "chunk_2": "evidence two",
            "chunk_3": "",
            "chunk_4": "",
        })
    rows.append({"example_id": "etest", "source_id": "stest", "split": "test", "claim": "test claim", "label": 0, "evidence_mask": [True, False, False, False], "chunk_1": "test evidence", "chunk_2": "", "chunk_3": "", "chunk_4": ""})
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "training_view_audit.json").write_text(json.dumps({"labels_used_for_claim_boundaries": False, "duplicate_groups_remaining": 0, "conflicting_duplicate_groups_remaining": 0, "input_mismatch_groups_included": 0, "split_source_overlap_count": 0}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": "ragtruth-qa-training-view-deduplicated-v1", "signature": "fixture", "artifacts": {"dataset.parquet": digest}}), encoding="utf-8")
    return path


def test_parquet_loader_has_fixed_four_slot_contract(tmp_path: Path) -> None:
    path = _parquet_fixture(tmp_path)
    rows, metadata = load_ragtruth_parquet(path, expected_signature="fixture")
    assert len(rows[0]["evidence"]) == 4
    assert rows[0]["evidence_mask"] == [True, True, False, False]
    assert rows[0]["label"] in {0, 1}
    assert metadata["dataset_format"] == "parquet"


def test_parquet_loader_rejects_nonempty_masked_slot(tmp_path: Path) -> None:
    path = _parquet_fixture(tmp_path)
    frame = pd.read_parquet(path)
    frame.loc[0, "chunk_3"] = "should be empty"
    frame.to_parquet(path, index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": "ragtruth-qa-training-view-deduplicated-v1", "signature": "fixture", "artifacts": {"dataset.parquet": hashlib.sha256(path.read_bytes()).hexdigest()}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mascarado"):
        load_ragtruth_parquet(path, expected_signature="fixture")


def test_group_split_is_deterministic_and_isolates_sources(tmp_path: Path) -> None:
    path = _parquet_fixture(tmp_path)
    rows, metadata = load_ragtruth_parquet(path, expected_signature="fixture")
    first = split_ragtruth_parquet(rows, metadata, split_seed=7)
    second = split_ragtruth_parquet(rows, metadata, split_seed=7)
    assert first.metadata["signature"] == second.metadata["signature"]
    assert first.assignments.equals(second.assignments)
    assert not ({row["source_id"] for row in first.train_rows} & {row["source_id"] for row in first.validation_rows})
    assert not ({row["source_id"] for row in first.test_rows} & {row["source_id"] for row in first.train_rows})


def test_collator_preserves_four_slots_and_boolean_mask() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.truncation = None

        def __call__(self, premises, claims, **kwargs):
            self.truncation = kwargs["truncation"]
            return {"input_ids": torch.ones((len(premises), 3), dtype=torch.long), "attention_mask": torch.ones((len(premises), 3), dtype=torch.long)}

    tokenizer = Tokenizer()
    collator = BagCollator(tokenizer, 8)
    batch = collator([
        {"example_id": "e", "source_id": "s", "task_type": "QA", "claim": "c", "evidence": ["a", "b", "", ""], "evidence_mask": [True, True, False, False], "label": 0}
    ])
    assert tuple(batch["input_ids"].shape) == (1, 4, 3)
    assert batch["evidence_mask"].dtype == torch.bool
    assert batch["evidence_mask"].tolist() == [[True, True, False, False]]
    assert tokenizer.truncation == "only_first"

    pt_tokenizer = Tokenizer()
    BagCollator(pt_tokenizer, 8, "longest_first")([
        {"example_id": "e", "source_id": "s", "task_type": "QA", "claim": "claim longo", "evidence": ["a", "b", "", ""], "evidence_mask": [True, True, False, False], "label": 0}
    ])
    assert pt_tokenizer.truncation == "longest_first"


def test_pt_nllb_uses_explicit_longest_first_without_changing_english_default() -> None:
    pt_config = load_config(Path("configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml"))
    english_config = load_config(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml"))
    assert pt_config.truncation == "longest_first"
    assert pt_config.to_dict()["truncation"] == "longest_first"
    assert english_config.truncation == "only_first"
    assert "truncation" not in english_config.to_dict()


def test_validate_data_only_config_does_not_require_model(tmp_path: Path) -> None:
    path = _parquet_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join([
        "run_name: fixture", "output_root: runs", "model_id: unavailable", "architecture: gated_attention", "encoder_mode: lora", "dataset:", "  format: parquet", f"  path: {path}", f"  manifest_path: {tmp_path / 'manifest.json'}", "  expected_signature: fixture", "  expected_schema: ragtruth-qa-training-view-deduplicated-v1", "training:", "  max_epochs: 1", "  planned_total_epochs: 1", "  class_weight:", "    mode: auto_pos_weight",
    ]), encoding="utf-8")
    config = load_config(config_path)
    result = validate_training_data(config)
    assert result["model_loaded"] is False
    assert result["forward_executed"] is False
    assert result["pos_weight"]["value"] is not None
