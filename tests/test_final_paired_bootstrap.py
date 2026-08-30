from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from ragtruth_transfer.factorial_paired_grouped_bootstrap import run_factorial_bootstrap, validate_factorial_frames
from ragtruth_transfer.final_paired_bootstrap import FinalBootstrapConfig, _metric_favors_lower, _significant
from ragtruth_transfer.paired_grouped_bootstrap import run_generic_pairwise_bootstrap, validate_generic_paired_frames
from ragtruth_transfer.thresholded_paired_grouped_bootstrap import _generic_summary, run_generic_thresholded_pair_bootstrap


def _frames() -> dict[str, dict[int, pd.DataFrame]]:
    base = pd.DataFrame({"example_id": ["a", "b", "c", "d"], "hearing_id": ["h1", "h1", "h2", "h2"], "label": [0, 1, 0, 1]})
    result: dict[str, dict[int, pd.DataFrame]] = {}
    for condition, offset in (("a", 0.0), ("b", 0.1), ("c", 0.2), ("d", 0.3)):
        result[condition] = {}
        for seed in (0, 1, 2):
            frame = base.copy()
            frame["probability"] = np.array([0.05, 0.45, 0.10, 0.40]) + offset + seed * 0.001
            result[condition][seed] = frame
    return result


def test_generic_pair_is_arbitrary_deterministic_and_preserves_alignment():
    frames = _frames()
    first = run_generic_pairwise_bootstrap(frames, {"b_minus_a": ("a", "b")}, seeds=(0, 1, 2), n_replicates=7,
                                           seed=19, expected_rows=4, expected_positives=2, expected_hearings=2)
    second = run_generic_pairwise_bootstrap(frames, {"b_minus_a": ("a", "b")}, seeds=(0, 1, 2), n_replicates=7,
                                            seed=19, expected_rows=4, expected_positives=2, expected_hearings=2)
    pd.testing.assert_frame_equal(first["replicates"], second["replicates"])
    assert (first["replicates"].loc[first["replicates"].metric != "brier", "delta"] >= 0).all()
    assert first["summaries"]["b_minus_a"]["brier"]["favorable_direction"] == "lower"


def test_generic_pair_rejects_exact_id_label_and_hearing_mismatches():
    frames = _frames()
    frames["b"][1] = frames["b"][1].assign(example_id=["a", "b", "c", "x"])
    with pytest.raises(ValueError, match="example_id mismatch"):
        validate_generic_paired_frames(frames, expected_rows=4, expected_positives=2, expected_hearings=2)
    frames = _frames()
    frames["b"][1] = frames["b"][1].assign(label=[0, 1, 1, 0])
    with pytest.raises(ValueError, match="label mismatch"):
        validate_generic_paired_frames(frames, expected_rows=4, expected_positives=2, expected_hearings=2)
    frames = _frames()
    frames["b"][1] = frames["b"][1].assign(hearing_id=["h2", "h1", "h2", "h2"])
    with pytest.raises(ValueError, match="hearing_id mismatch"):
        validate_generic_paired_frames(frames, expected_rows=4, expected_positives=2, expected_hearings=2)


def test_generic_thresholded_arbitrary_pair_freezes_direction_and_is_deterministic():
    frames = _frames()
    thresholds = {condition: {seed: {"best_f1": 0.3, "fpr10": 0.3} for seed in (0, 1, 2)} for condition in frames}
    first = run_generic_thresholded_pair_bootstrap(frames, thresholds, "a", "b", n_replicates=7, seed=19, expected_rows=4, expected_positives=2, expected_hearings=2)
    second = run_generic_thresholded_pair_bootstrap(frames, thresholds, "a", "b", n_replicates=7, seed=19, expected_rows=4, expected_positives=2, expected_hearings=2)
    pd.testing.assert_frame_equal(first["replicates"], second["replicates"])
    assert first["summaries"]["best_f1"]["fpr"]["bootstrap"]["favorable_direction"] == "lower"


def test_factorial_arbitrary_registry_and_interaction_use_shared_bootstrap():
    frames = _frames()
    result = run_factorial_bootstrap(frames, conditions=("a", "b", "c", "d"),
                                     contrasts={"b_minus_a": ("a", "b"), "d_minus_c": ("c", "d")},
                                     interactions={"interaction": ("b_minus_a", "d_minus_c")}, metrics=("auprc",),
                                     expected_rows=4, expected_positives=2, expected_hearings=2, n_replicates=7, seed=19)
    assert "interaction" in result["summaries"]
    assert result["protocol"]["same_samples_all_conditions"] is True
    assert result["summaries"]["interaction"]["auprc"]["n_valid"] == 7
    arbitrary = validate_factorial_frames(frames, conditions=("a", "b", "c", "d"), expected_rows=4, expected_positives=2, expected_hearings=2)
    assert arbitrary["exact_example_alignment"] is True


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([0.01, 0.02, 0.03]), "FAVOR_A"),
        (np.array([-0.03, -0.02, -0.01]), "FAVOR_B"),
        (np.array([-0.01, 0.01]), "INCONCLUSIVE"),
    ],
)
def test_fpr_orientation_uses_lower_is_better_for_verdict_and_probability(values, expected):
    summary = _generic_summary(values, 0.95, favorable_lower=True)
    assert _metric_favors_lower("fpr") is True
    assert _significant(summary, "fpr") == expected
    expected_probability = float(np.mean(values < 0))
    assert summary["probability_favorable"] == expected_probability


def test_brier_remains_lower_is_better_and_higher_metrics_remain_unchanged():
    negative = {"ci_lower": -0.03, "ci_upper": -0.01}
    positive = {"ci_lower": 0.01, "ci_upper": 0.03}
    crossing = {"ci_lower": -0.01, "ci_upper": 0.01}
    assert _metric_favors_lower("brier") is True
    assert _significant(negative, "brier") == "FAVOR_B"
    assert _significant(positive, "brier") == "FAVOR_A"
    for metric in ("auprc", "auroc", "precision", "recall", "f1", "mcc"):
        assert _metric_favors_lower(metric) is False
        assert _significant(positive, metric) == "FAVOR_B"
        assert _significant(negative, metric) == "FAVOR_A"
        assert _significant(crossing, metric) == "INCONCLUSIVE"


def test_set_effect_interactions_hold_en_training_constant():
    config = FinalBootstrapConfig.from_yaml(Path("configs/publichearing_final_paired_bootstrap.yaml"))
    assert config.interactions["nllb_set_effect_target_minus_pt"] == ("en_set_vs_attention_ph_en_nllb", "en_set_vs_attention_ph_pt")
    assert config.interactions["madlad_set_effect_target_minus_pt"] == ("en_set_vs_attention_ph_en_madlad", "en_set_vs_attention_ph_pt")
