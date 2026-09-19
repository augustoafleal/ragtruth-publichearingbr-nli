from dataclasses import asdict
from pathlib import Path

from ragtruth_transfer.publichearing.campaign import _experiment_identifier, _stable_hash
from ragtruth_transfer.publichearing.config import load_publichearing_config


ROOT = Path(__file__).resolve().parents[1]


def _signature(config) -> str:
    return _stable_hash(
        {
            "config": config.to_dict(),
            "dataset_sha256": "dataset-sha256",
            "eligible_sample_ids": ["sample-1", "sample-2"],
            "splits_sha256": "splits-sha256",
        }
    )


def test_historical_attention_config_keeps_signature_representation() -> None:
    config = load_publichearing_config(
        ROOT / "configs/publichearing_lora_attention_mil_confirmatory.yaml"
    )
    historical = asdict(config)
    historical.pop("architecture")
    historical.pop("set_transformer")

    assert config.to_dict() == historical
    assert _signature(config) == _stable_hash(
        {
            "config": historical,
            "dataset_sha256": "dataset-sha256",
            "eligible_sample_ids": ["sample-1", "sample-2"],
            "splits_sha256": "splits-sha256",
        }
    )


def test_set_transformer_has_distinct_signature_and_provenance() -> None:
    attention = load_publichearing_config(
        ROOT / "configs/publichearing_lora_attention_mil_confirmatory.yaml"
    )
    set_transformer = load_publichearing_config(
        ROOT / "configs/publichearing_lora_set_transformer_mil_confirmatory.yaml"
    )

    assert _signature(set_transformer) != _signature(attention)
    assert set_transformer.to_dict()["architecture"] == "set_transformer"
    assert "set_transformer" in set_transformer.to_dict()
    assert _experiment_identifier(attention) == "publichearingbr_lora_attention_mil_5fold"
    assert _experiment_identifier(set_transformer) == "publichearingbr_lora_set_transformer_mil_5fold"
