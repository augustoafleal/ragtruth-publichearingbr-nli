from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.io_utils import sha256_file
from ragtruth_transfer.set_transformer_paired_grouped_bootstrap import (
    CampaignInput,
    SetTransformerBootstrapConfig,
    _campaign_pooling_type,
    _generate_replicates,
    _input_signature,
    _strict_join_campaigns,
    _verify_completed,
    _validate_prediction_seed_column,
)


def _config(tmp_path: Path, *, n_replicates: int = 8, seed: int = 20260815) -> SetTransformerBootstrapConfig:
    return SetTransformerBootstrapConfig(
        output_root=tmp_path / "out",
        campaigns={
            "gated_attention": CampaignInput("gated_attention", tmp_path / "gated", "4e12933c51136624", "probability"),
            "set_transformer": CampaignInput("set_transformer", tmp_path / "set", "28323e6cca11feb6", "probability"),
        },
        comparisons=(("set_transformer", "gated_attention"),),
        seeds=(0, 1, 2),
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=0.95,
        ci_method="percentile",
        preserve_group_multiplicity=True,
        same_samples_across_methods=True,
        same_samples_across_seeds=True,
        metrics=("auprc", "auroc", "brier"),
    )


def _frame(values: list[float], *, seed: int | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d", "e", "f"],
            "hearing_id": ["h1", "h1", "h2", "h2", "h3", "h3"],
            "label": [0, 1, 0, 1, 0, 1],
            "probability": values,
            "publichearing_dataset_signature": ["public"] * 6,
            "model_run_signature": ["run"] * 6,
        }
    )
    if seed is not None:
        frame["seed"] = np.asarray([seed] * len(frame), dtype=np.int64)
    return frame


def test_join_pairs_exact_examples_hearings_and_accepts_historical_gated_without_seed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frames = {
        "gated_attention": {seed: _frame([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]) for seed in config.seeds},
        "set_transformer": {seed: _frame([0.2, 0.8, 0.3, 0.7, 0.4, 0.6], seed=seed) for seed in config.seeds},
    }
    joined, audit, public_signature = _strict_join_campaigns(frames, config)
    assert len(joined) == 6
    assert audit["gated_attention"]["0"]["seed_identification"] == "historical_path"
    assert audit["set_transformer"]["2"]["seed_identification"] == "explicit"
    assert public_signature == "public"

    frames["set_transformer"][1].loc[0, "hearing_id"] = "other"
    with pytest.raises(ValueError, match="não pareáveis"):
        _strict_join_campaigns(frames, config)


def test_join_rejects_label_or_example_divergence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frames = {
        "gated_attention": {seed: _frame([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]) for seed in config.seeds},
        "set_transformer": {seed: _frame([0.2, 0.8, 0.3, 0.7, 0.4, 0.6], seed=seed) for seed in config.seeds},
    }
    frames["set_transformer"][0].loc[0, "label"] = 1
    with pytest.raises(ValueError, match="não pareáveis"):
        _strict_join_campaigns(frames, config)

    frames["set_transformer"][0] = frames["set_transformer"][0].iloc[:-1]
    with pytest.raises(ValueError, match="não pareáveis"):
        _strict_join_campaigns(frames, config)


def test_set_seed_column_is_integer_and_matches_run_seed(tmp_path: Path) -> None:
    campaign = CampaignInput("set_transformer", tmp_path / "set", "28323e6cca11feb6", "probability")
    _validate_prediction_seed_column(_frame([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], seed=1), campaign, 1)
    with pytest.raises(ValueError, match="inteira"):
        bad = _frame([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], seed=1)
        bad["seed"] = bad["seed"].astype(float)
        _validate_prediction_seed_column(bad, campaign, 1)
    with pytest.raises(ValueError, match="divergente"):
        _validate_prediction_seed_column(_frame([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], seed=2), campaign, 1)


def test_historical_alias_normalization_is_local(tmp_path: Path) -> None:
    gated = CampaignInput("gated_attention", tmp_path / "gated", "4e12933c51136624", "probability")
    gated.path.mkdir()
    (gated.path / "resolved_config.json").write_text(json.dumps({"architecture": "gated_attention"}), encoding="utf-8")
    assert _campaign_pooling_type(gated, {}) == "gated_attention"
    (gated.path / "resolved_config.json").write_text(json.dumps({"architecture": "attention"}), encoding="utf-8")
    assert _campaign_pooling_type(gated, {"pooling_type": "attention"}) == "gated_attention"


def test_replicates_preserve_groups_use_shared_samples_and_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path, n_replicates=12)
    joined = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d", "e", "f"],
            "hearing_id": ["h1", "h1", "h2", "h2", "h3", "h3"],
            "label": [0, 1, 0, 1, 0, 1],
            **{f"{name}_seed_{seed}": values for name, values in (("gated_attention", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]), ("set_transformer", [0.2, 0.8, 0.3, 0.7, 0.4, 0.6])) for seed in config.seeds},
        }
    )
    first = _generate_replicates(joined, config)
    second = _generate_replicates(joined, config)
    pd.testing.assert_frame_equal(first, second)
    assert first["n_sampled_groups"].eq(3).all()
    assert first.groupby("replicate_id")["sampled_groups_hash"].nunique().eq(1).all()
    assert first.groupby(["replicate_id", "metric"]).size().eq(4).all()
    assert first.loc[first["seed"] == -1, "seed_label"].eq("mean_seed_delta").all()


def test_delta_sign_and_brier_interpretation(tmp_path: Path) -> None:
    config = _config(tmp_path, n_replicates=3)
    joined = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "hearing_id": ["h1", "h1", "h2", "h2"],
            "label": [0, 1, 0, 1],
            **{f"gated_attention_seed_{seed}": [0.1, 0.9, 0.2, 0.8] for seed in config.seeds},
            **{f"set_transformer_seed_{seed}": [0.9, 0.1, 0.8, 0.2] for seed in config.seeds},
        }
    )
    replicates = _generate_replicates(joined, config)
    assert (replicates.loc[replicates["metric"] == "auprc", "delta"].dropna() < 0).all()
    assert (replicates.loc[replicates["metric"] == "brier", "delta"].dropna() > 0).all()


def test_input_signature_changes_with_input_hashes_but_not_output_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    joined = pd.DataFrame({"example_id": ["a"], "hearing_id": ["h"]})
    hashes = {f"{name}_seed_{seed}_predictions.parquet": f"{name}-{seed}" for name in config.campaigns for seed in config.seeds}
    signature = _input_signature(config, hashes, joined, "public")[0]
    changed = dict(hashes)
    changed["set_transformer_seed_1_predictions.parquet"] = "changed"
    assert signature != _input_signature(config, changed, joined, "public")[0]
    assert signature == _input_signature(replace(config, output_root=tmp_path / "other"), hashes, joined, "public")[0]


def test_completed_manifest_resume_validates_artifact_hashes(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    artifact = output / "comparison_summary.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    manifest = {
        "status": "completed",
        "signature": "abc123",
        "artifacts": {artifact.name: sha256_file(artifact)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _verify_completed(output, "abc123")["status"] == "completed"
    artifact.write_text('{"ok": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="corrompido"):
        _verify_completed(output, "abc123")


def test_config_serialization_declares_fixed_comparison() -> None:
    config = SetTransformerBootstrapConfig.from_yaml(Path("configs/publichearing_set_transformer_paired_grouped_bootstrap.yaml"))
    serialized = config.to_dict()
    assert config.seeds == (0, 1, 2)
    assert config.n_replicates == 10_000
    assert serialized["comparisons"] == [{"left": "set_transformer", "right": "gated_attention", "delta_definition": "left - right"}]
    assert "negative brier favors Set Transformer" in serialized["delta_convention"]
