from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ragtruth_transfer.paired_grouped_bootstrap import BootstrapConfig, _build_summaries, _effect, _input_signature, _metric_value, _resume_from_complete_replicates, _strict_join, _validate_prediction


def _frame(ids=("a", "b", "c", "d"), labels=(0, 1, 0, 1), hearings=("h1", "h1", "h2", "h2"), score_name="hallucination_score"):
    return pd.DataFrame({"example_id": list(ids), "hearing_id": list(hearings), "label": list(labels), score_name: [0.1, 0.9, 0.2, 0.8]})


def test_strict_join_rejects_missing_and_label_or_group_mismatch():
    base = _frame()
    good = _frame(score_name="probability")
    joined, audit = _strict_join(base, {0: good, 1: good.copy(), 2: good.copy()})
    assert len(joined) == 4 and audit["0"]["pairable_examples"] == 4
    with pytest.raises(ValueError):
        bad_ids = _frame(ids=("a", "b", "c", "x"), score_name="probability")
        _strict_join(base, {0: bad_ids, 1: good.copy(), 2: good.copy()})
    bad = _frame(score_name="probability"); bad.loc[0, "label"] = 1
    with pytest.raises(ValueError):
        _strict_join(base, {0: bad, 1: good.copy(), 2: good.copy()})
    bad = _frame(score_name="probability"); bad.loc[0, "hearing_id"] = "other"
    with pytest.raises(ValueError):
        _strict_join(base, {0: bad, 1: good.copy(), 2: good.copy()})


def test_effect_orientations_and_metric_values():
    labels = np.array([0, 1, 0, 1])
    baseline = np.array([0.1, 0.2, 0.2, 0.4])
    model = np.array([0.1, 0.8, 0.2, 0.9])
    assert _metric_value("auprc", labels, model) >= _metric_value("auprc", labels, baseline)
    assert np.isclose(_effect("auprc", 0.4, 0.6), 0.2)
    assert np.isclose(_effect("auroc", 0.4, 0.6), 0.2)
    assert np.isclose(_effect("brier_improvement", 0.2, 0.1), 0.1)


def test_group_bootstrap_indices_preserve_repeated_group_multiplicity():
    groups = np.array(["h1", "h1", "h2"])
    group_indices = [np.flatnonzero(groups == group) for group in sorted(np.unique(groups))]
    sampled_positions = np.array([0, 0])
    sampled = np.concatenate([group_indices[position] for position in sampled_positions])
    assert sampled.tolist() == [0, 1, 0, 1]
    assert len(sampled) == 4 and len(set(sampled.tolist())) == 2


def _complete_frame(score_name="hallucination_score"):
    return pd.DataFrame({"example_id": [f"e{i}" for i in range(4235)], "hearing_id": [f"h{i % 206}" for i in range(4235)], "label": [1] * 501 + [0] * 3734, score_name: np.linspace(0.01, 0.99, 4235)})


def test_input_validation_rejects_nonbinary_labels_duplicate_and_invalid_scores(tmp_path):
    frame = _complete_frame()
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)
    assert len(_validate_prediction(path, "hallucination_score")) == 4235
    bad = frame.copy(); bad["label"] = bad["label"].astype(float); bad.loc[0, "label"] = 0.5; bad.to_parquet(path, index=False)
    with pytest.raises(ValueError): _validate_prediction(path, "hallucination_score")
    bad = frame.copy(); bad.loc[1, "example_id"] = bad.loc[0, "example_id"]; bad.to_parquet(path, index=False)
    with pytest.raises(ValueError): _validate_prediction(path, "hallucination_score")
    for value in (np.nan, np.inf, 1.1):
        bad = frame.copy(); bad.loc[0, "hallucination_score"] = value; bad.to_parquet(path, index=False)
        with pytest.raises(ValueError): _validate_prediction(path, "hallucination_score")


def test_summary_uses_valid_effects_and_keeps_invalid_replicates_out_of_metrics():
    config = replace(BootstrapConfig.from_yaml(Path("configs/publichearing_paired_grouped_bootstrap.yaml")), n_replicates=2)
    rows = []
    for replicate_id, valid, effect in ((0, True, 0.2), (1, False, None)):
        for seed in (0, 1, 2, -1):
            for metric in config.metrics:
                rows.append({"replicate_id": replicate_id, "bootstrap_seed": config.seed, "seed": seed, "metric": metric, "valid": valid, "invalid_reason": None if valid else "single_class_resample", "effect": effect})
    observed = {"baseline": {"auprc": 0.2, "auroc": 0.3, "brier": 0.4}, **{f"seed_{seed}": {"auprc": 0.4, "auroc": 0.5, "brier": 0.2} for seed in (0, 1, 2)}}
    effects = {str(seed): {"auprc": 0.2, "auroc": 0.2, "brier_improvement": 0.2} for seed in (0, 1, 2)}
    per_seed, campaign, valid, invalid, reasons = _build_summaries(pd.DataFrame(rows), config, observed, effects)
    assert valid == 1 and invalid == 1
    assert reasons == {"single_class_resample": 1}
    assert per_seed["0"]["bootstrap"]["auprc"]["n_valid"] == 1
    assert campaign["bootstrap"]["auroc"]["bootstrap_support_probability"] == 1.0


def test_signature_excludes_absolute_paths_but_changes_with_input_hashes():
    config = BootstrapConfig.from_yaml(Path("configs/publichearing_paired_grouped_bootstrap.yaml"))
    joined = pd.DataFrame({"example_id": ["a"], "hearing_id": ["h"]})
    hashes = {"baseline_predictions.parquet": "a", "seed_0_predictions.parquet": "b", "seed_1_predictions.parquet": "c", "seed_2_predictions.parquet": "d"}
    signature = _input_signature(config, hashes, joined)[0]
    moved = replace(config, output_root=Path("/tmp/output"), baseline_run=Path("/tmp/base"), confirmatory_run=Path("/tmp/confirm"))
    assert signature == _input_signature(moved, hashes, joined)[0]
    changed = dict(hashes); changed["seed_0_predictions.parquet"] = "changed"
    assert signature != _input_signature(config, changed, joined)[0]


def test_resume_rebuilds_only_derived_artifacts_from_complete_replicates(tmp_path):
    config = replace(BootstrapConfig.from_yaml(Path("configs/publichearing_paired_grouped_bootstrap.yaml")), n_replicates=1)
    rows = []
    for seed in (0, 1, 2, -1):
        for metric in config.metrics:
            rows.append({"replicate_id": 0, "bootstrap_seed": config.seed, "seed": seed, "metric": metric, "baseline_value": 0.2, "zero_shot_value": 0.4 if seed >= 0 else None, "effect": 0.2, "valid": True, "invalid_reason": None, "n_rows": 4, "n_sampled_groups": 2, "n_unique_groups": 1, "n_positive": 2, "n_negative": 2})
    pd.DataFrame(rows).to_parquet(tmp_path / "bootstrap_replicates.parquet", index=False)
    observed = {"baseline": {"auprc": 0.2, "auroc": 0.3, "brier": 0.4}, **{f"seed_{seed}": {"auprc": 0.4, "auroc": 0.5, "brier": 0.2} for seed in (0, 1, 2)}}
    effects = {str(seed): {"auprc": 0.2, "auroc": 0.2, "brier_improvement": 0.2} for seed in (0, 1, 2)}
    manifest = _resume_from_complete_replicates(tmp_path, config, "sig", {"protocol": "x"}, observed, effects, {"baseline_predictions.parquet": "x"}, {"0": {"pairable_examples": 4}}, 4, 2, 2)
    assert manifest["status"] == "completed"
    assert (tmp_path / "campaign_summary.json").is_file()
