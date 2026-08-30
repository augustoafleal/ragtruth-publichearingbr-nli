from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.thresholded_paired_grouped_bootstrap import (
    GenericThresholdedBootstrapConfig,
    _generic_strict_join,
    _generic_summary,
    load_thresholded_config,
    run_generic_thresholded_bootstrap,
)


CONFIG_PATH = Path("configs/publichearing_pt_nllb_attention_vs_set_thresholded_bootstrap.yaml")


def _require_frozen_campaigns() -> GenericThresholdedBootstrapConfig:
    config = load_thresholded_config(CONFIG_PATH)
    assert isinstance(config, GenericThresholdedBootstrapConfig)
    missing = [str(spec.path / "manifest.json") for spec in (config.attention, config.set_transformer) if not (spec.path / "manifest.json").is_file()]
    if missing:
        pytest.skip(f"requires frozen PT-NLLB campaign artifacts: {', '.join(missing)}")
    return config


@pytest.mark.integration
def test_attention_and_set_campaigns_are_accepted_by_preflight():
    config = _require_frozen_campaigns()
    result = run_generic_thresholded_bootstrap(config, validate_only=True)
    assert result["status"] == "valid"
    assert result["n_examples"] == 4235
    assert result["n_hearings"] == 206
    assert result["thresholds_frozen"] is True
    assert result["threshold_audit"]["publichearing_labels_used_for_threshold_selection"] is False


def test_wrong_pooling_is_rejected():
    config = load_thresholded_config(CONFIG_PATH)
    assert isinstance(config, GenericThresholdedBootstrapConfig)
    bad = replace(config, attention=replace(config.attention, expected_pooling="set_transformer"))
    with pytest.raises(ValueError, match="Attention MIL contra Set Transformer"):
        bad.validate({"analysis_type": "thresholded_paired_grouped_bootstrap", "group_key": "hearing_id", "paired": True})


def test_alignment_rejects_label_or_hearing_mismatch():
    config = load_thresholded_config(CONFIG_PATH)
    assert isinstance(config, GenericThresholdedBootstrapConfig)
    config = replace(config, expected_examples=4, expected_positives=2, expected_hearings=2)
    base = pd.DataFrame({"example_id": ["a", "b", "c", "d"], "hearing_id": ["h1", "h1", "h2", "h2"], "label": [0, 1, 0, 1], "probability": [.1, .9, .2, .8]})
    frames = {"attention": {0: base, 1: base.copy(), 2: base.copy()}, "set_transformer": {0: base.copy(), 1: base.copy(), 2: base.copy()}}
    joined, audit = _generic_strict_join(frames, config)
    assert len(joined) == 4 and all(item["pairable_examples"] == 4 for model in audit.values() for item in model.values())
    bad = {model: {seed: frame.copy() for seed, frame in values.items()} for model, values in frames.items()}
    bad["set_transformer"][1].loc[0, "label"] = 1
    with pytest.raises(ValueError, match="Pareamento estrito"):
        _generic_strict_join(bad, config)
    bad = {model: {seed: frame.copy() for seed, frame in values.items()} for model, values in frames.items()}
    bad["set_transformer"][2].loc[0, "hearing_id"] = "other"
    with pytest.raises(ValueError, match="Pareamento estrito"):
        _generic_strict_join(bad, config)


def test_summary_uses_lower_is_better_direction_for_fpr():
    values = np.array([-0.2, -0.1, 0.1, 0.2])
    assert _generic_summary(values, 0.95, favorable_lower=True)["favorable_probability"] == 0.5
    assert _generic_summary(values, 0.95, favorable_lower=False)["favorable_probability"] == 0.5


@pytest.mark.integration
def test_small_real_bootstrap_is_deterministic_and_thresholds_are_fixed(tmp_path):
    config = _require_frozen_campaigns()
    small = replace(config, n_replicates=3, output_root=tmp_path / "analysis_a")
    first = run_generic_thresholded_bootstrap(small)
    second = run_generic_thresholded_bootstrap(replace(small, output_root=tmp_path / "analysis_b"))
    first_frame = pd.read_parquet(tmp_path / "analysis_a" / first["signature"] / "bootstrap_replicates.parquet")
    second_frame = pd.read_parquet(tmp_path / "analysis_b" / second["signature"] / "bootstrap_replicates.parquet")
    pd.testing.assert_frame_equal(first_frame, second_frame, check_categorical=False)
    assert first_frame.groupby(["regime", "seed"])[["threshold_attention", "threshold_set_transformer"]].nunique().to_numpy().max() == 1
