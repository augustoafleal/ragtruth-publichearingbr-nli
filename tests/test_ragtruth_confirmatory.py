from __future__ import annotations

from pathlib import Path

from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig, validate_only


def test_confirmatory_protocol_is_three_seed_and_split_is_fixed() -> None:
    config = ConfirmatoryConfig.from_yaml(Path("configs/ragtruth_lora_attention_mil_confirmatory.yaml"))
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
