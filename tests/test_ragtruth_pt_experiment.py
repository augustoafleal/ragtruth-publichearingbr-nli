from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig
from ragtruth_transfer.training import prepare_training_data
from scripts.filter_ragtruth_translation_failures import filter_dataset, validate_filtered_dataset
from scripts.run_ragtruth_pt_nllb_experiment import main


PT_CONFIG = Path("configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml")
FILTERED_DIR = Path("data/processed/ragtruth_textual_pt_nllb_filtered")


def test_pt_config_selects_only_filtered_jsonl_and_is_output_isolated() -> None:
    config = ConfirmatoryConfig.from_yaml(PT_CONFIG)
    dataset = config.experiment.dataset
    assert dataset.format == "jsonl"
    assert dataset.path == FILTERED_DIR.resolve()
    assert dataset.path != Path("data/processed/ragtruth_textual_pt_nllb").resolve()
    assert dataset.path != Path("data/processed/ragtruth_textual").resolve()
    assert config.experiment.output_root != dataset.path
    assert config.experiment.output_root == Path("runs/ragtruth_pt_nllb_filtered_confirmatory").resolve()
    assert config.experiment.training.planned_total_epochs == 6
    assert config.experiment.training.gradient_accumulation_steps == 16
    assert config.experiment.training.task_balanced_sampler is True
    assert config.seeds == (0, 1, 2)
    assert config.experiment.dataset.evaluate_test is False


@pytest.mark.integration
def test_pt_jsonl_split_loading_has_frozen_counts() -> None:
    if not FILTERED_DIR.is_dir():
        pytest.skip("requires the generated filtered PT dataset")
    config = ConfirmatoryConfig.from_yaml(PT_CONFIG)
    train, validation, test, metadata, hashes = prepare_training_data(
        config.experiment, config.experiment.dataset.path
    )
    assert (len(train), len(validation), len(test)) == (45238, 8143, 9121)
    assert metadata["schema_version"] == "ragtruth-translated-filtered-v1"
    assert hashes["split"] == config.expected_split_signature


def test_filtered_manifest_missing_or_malformed_fails_fast(tmp_path: Path) -> None:
    output = tmp_path / "filtered"
    output.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_filtered_dataset(
            root=tmp_path,
            source_dir=tmp_path / "source",
            translated_dir=tmp_path / "translated",
            output_dir=output,
            qa_config={},
        )

    (output / "manifest.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed filtered manifest"):
        validate_filtered_dataset(
            root=tmp_path,
            source_dir=tmp_path / "source",
            translated_dir=tmp_path / "translated",
            output_dir=output,
            qa_config={},
        )


def test_filtered_manifest_missing_split_fails_fast(tmp_path: Path) -> None:
    source = tmp_path / "source"
    translated = tmp_path / "translated"
    output = tmp_path / "filtered"
    source.mkdir()
    translated.mkdir()
    row = {
        "example_id": "one",
        "source_id": "source-one",
        "label": False,
        "claim": "The proposal is clear",
        "evidence": ["The answer is clear"],
        "evidence_mask": [True],
    }
    for directory in (source, translated):
        for split in ("train", "validation", "test"):
            (directory / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    filter_dataset(
        root=tmp_path,
        source_dir=source,
        translated_dir=translated,
        output_dir=output,
        qa_config={"expected_candidate_translation_pairs": 0},
    )
    (output / "validation.jsonl").unlink()
    with pytest.raises(ValueError, match="missing or hash mismatch"):
        validate_filtered_dataset(
            root=tmp_path,
            source_dir=source,
            translated_dir=translated,
            output_dir=output,
            qa_config={"expected_candidate_translation_pairs": 0},
        )


@pytest.mark.integration
def test_pt_wrapper_dry_run_does_not_train(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    if not FILTERED_DIR.is_dir():
        pytest.skip("requires the generated filtered PT dataset")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ragtruth_pt_nllb_experiment.py", "--config", str(PT_CONFIG), "--dry-run"],
    )
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run_valid"
    assert result["filter_action"] == "reused"
    assert result["training_validation"]["model_loaded"] is False
    assert result["training_validation"]["training_executed"] is False
    assert result["training_validation"]["data_audit"]["partitions"]["train"]["examples"] == 45238
