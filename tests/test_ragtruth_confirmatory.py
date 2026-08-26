from __future__ import annotations

from pathlib import Path

import pytest

from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig, validate_only


def test_confirmatory_protocol_declares_three_fixed_seeds() -> None:
    config = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml"))
    assert config.seeds == (0, 1, 2)
    assert config.split_seed == 42
    assert config.expected_dataset_sha256 == "357e05b08cdcc22b766dce432fd8ed5caa7703ddf144dc02da24ef63e7ff0a7c"
    assert config.expected_split_signature == "525edec2966a4fac"


def test_madlad_confirmatory_config_declares_pending_translated_dataset_contract() -> None:
    config = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_pt_madlad_lora_attention_mil_confirmatory.yaml"))
    assert config.experiment.dataset.path is not None
    assert config.experiment.dataset.path.name == "dataset.parquet"
    assert config.experiment.dataset.expected_signature is None
    assert config.expected_dataset_signature is None
    assert config.experiment.truncation == "longest_first"
    assert config.zero_shot_config_path is not None
    assert config.zero_shot_config_path.name == "ragtruth_pt_madlad_to_publichearing_zero_shot.yaml"


@pytest.mark.integration
def test_confirmatory_protocol_validates_frozen_training_view_when_available() -> None:
    config = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml"))
    dataset_path = config.experiment.dataset.path
    manifest_path = config.experiment.dataset.manifest_path
    if dataset_path is None or manifest_path is None or not dataset_path.is_file() or not manifest_path.is_file():
        pytest.skip("requires the unversioned frozen RAGTruth training view")
    result = validate_only(config)
    assert result["status"] == "valid"
    assert result["model_loaded"] is False
    assert result["seeds"] == [0, 1, 2]
    assert result["data_audit"]["metadata"]["dataset_sha256"] == "357e05b08cdcc22b766dce432fd8ed5caa7703ddf144dc02da24ef63e7ff0a7c"
    assert result["data_audit"]["split"]["signature"] == "525edec2966a4fac"


def test_confirmatory_signature_changes_with_protocol_seed() -> None:
    config = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml"))
    payload = config.protocol_payload("525edec2966a4fac")
    from ragtruth_transfer.ragtruth_confirmatory import _signature
    first = _signature(payload)
    changed = dict(payload)
    changed["bootstrap_seed"] = 99
    assert _signature(changed) != first


def test_madlad_target_evaluation_configs_are_target_aware() -> None:
    attention = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_en_attention_to_publichearing_en_madlad.yaml"))
    set_transformer = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_en_set_to_publichearing_en_madlad.yaml"))
    assert attention.evaluation_output_root is not None
    assert set_transformer.evaluation_output_root is not None
    assert attention.evaluation_output_root != set_transformer.evaluation_output_root
    assert attention.zero_shot_config_path is not None
    assert set_transformer.zero_shot_config_path is not None
    from ragtruth_transfer.ragtruth_zero_shot import load_zero_shot_config
    attention_target = load_zero_shot_config(attention.zero_shot_config_path)
    set_target = load_zero_shot_config(set_transformer.zero_shot_config_path)
    assert attention_target.publichearing_target_id == "publichearing_en_madlad"
    assert set_target.publichearing_target_id == "publichearing_en_madlad"
    assert attention_target.publichearing_path.name == "PublicHearingBR_NLI.jsonl"
    assert "publichearing_nli_pt_to_en_madlad" in str(attention_target.publichearing_path)
