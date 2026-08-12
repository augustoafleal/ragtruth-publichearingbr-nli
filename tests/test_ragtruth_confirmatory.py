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
