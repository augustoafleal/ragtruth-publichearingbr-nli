from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ragtruth_transfer.pooling_paired_grouped_bootstrap import (
    PoolingBootstrapConfig,
    _input_signature,
    _observed_metrics,
    _strict_join_campaigns,
)


def _frame(probabilities: list[float], signature: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "hearing_id": ["h1", "h1", "h2", "h2"],
            "label": [0, 1, 0, 1],
            "probability": probabilities,
            "publichearing_dataset_signature": ["public"] * 4,
            "model_run_signature": [signature] * 4,
        }
    )


def test_pooling_join_requires_exact_example_labels_and_hearings():
    config = PoolingBootstrapConfig.from_yaml(Path("configs/publichearing_pooling_paired_grouped_bootstrap.yaml"))
    frames = {
        name: {seed: _frame([0.1, 0.9, 0.2, 0.8], campaign.signature) for seed in config.seeds}
        for name, campaign in config.campaigns.items()
    }
    joined, audit, public_signature = _strict_join_campaigns(frames, config)
    assert len(joined) == 4
    assert audit["max"]["2"]["pairable_examples"] == 4
    assert public_signature == "public"
    frames["mean"][1].loc[0, "hearing_id"] = "other"
    with pytest.raises(ValueError, match="não pareáveis"):
        _strict_join_campaigns(frames, config)


def test_pooling_observed_deltas_use_left_minus_right_for_all_metrics():
    config = PoolingBootstrapConfig.from_yaml(Path("configs/publichearing_pooling_paired_grouped_bootstrap.yaml"))
    joined = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            **{
                f"{name}_seed_{seed}": ([0.1, 0.9, 0.2, 0.8] if name == "gated_attention" else [0.4, 0.6, 0.4, 0.6])
                for name in config.campaigns
                for seed in config.seeds
            },
        }
    )
    observed, comparisons = _observed_metrics(joined.label.to_numpy(), joined, config)
    comparison = comparisons["gated_attention_vs_max"]
    for metric in config.metrics:
        assert comparison["mean_delta_across_seeds"][metric] == pytest.approx(
            observed["gated_attention"]["mean_across_seeds"][metric] - observed["max"]["mean_across_seeds"][metric]
        )
    assert comparison["mean_delta_across_seeds"]["brier"] < 0


def test_pooling_signature_changes_when_any_campaign_prediction_changes():
    config = PoolingBootstrapConfig.from_yaml(Path("configs/publichearing_pooling_paired_grouped_bootstrap.yaml"))
    joined = pd.DataFrame({"example_id": ["a"], "hearing_id": ["h"]})
    hashes = {f"{name}_seed_{seed}_predictions.parquet": f"{name}-{seed}" for name in config.campaigns for seed in config.seeds}
    signature = _input_signature(config, hashes, joined, "public")[0]
    changed = dict(hashes)
    changed["max_seed_1_predictions.parquet"] = "changed"
    assert signature != _input_signature(config, changed, joined, "public")[0]
    assert signature == _input_signature(replace(config, output_root=Path("/tmp/other")), hashes, joined, "public")[0]
