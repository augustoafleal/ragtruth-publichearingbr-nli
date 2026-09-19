from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from ragtruth_transfer.translation_quality import linkage
from ragtruth_transfer.translation_quality.config import LinkConfig
from ragtruth_transfer.translation_quality.linkage import (
    LinkValidationError,
    add_error_columns,
    link_quality_to_detection,
    load_protocol_threshold,
)

PROTOCOL_SIGNATURE = "70bb1cce59b8c824"
FROZEN_THRESHOLD = 0.42


def _synthetic_frames():
    scores_rows = []
    preds_rows = []
    sids = ["s0", "s1", "s2", "s3"]
    idx = 0
    for bucket in range(4):
        sep = bucket / 3.0
        for k in range(16):
            label = k % 2
            score = 0.5 + 0.45 * sep if label else 0.5 - 0.45 * sep
            score = min(0.999, max(0.001, score))
            example_id = f"ex-{idx}"
            scores_rows.append(
                {
                    "example_id": example_id,
                    "source_id": sids[k % 4],
                    "label": bool(label),
                    "nli_abs_delta_max": 0.9 - 0.2 * bucket + 0.005 * k,
                    "nli_abs_delta_mean": 0.8 - 0.2 * bucket,
                    "nli_any_flip": bucket == 0,
                    "any_truncation_introduced": bucket == 0,
                }
            )
            preds_rows.append(
                {"example_id": example_id, "source_id": sids[k % 4], "score": float(score), "label": bool(label)}
            )
            idx += 1
    return pd.DataFrame(scores_rows), pd.DataFrame(preds_rows)


def _write_inputs(tmp_path, *, include_thresholds: bool = True):
    seed_dir = tmp_path / "runs" / "ragtruth_pt_nllb_confirmatory" / PROTOCOL_SIGNATURE / "seed_0"
    test_dir = seed_dir / "ragtruth_test"
    test_dir.mkdir(parents=True)
    thresholds_path = seed_dir / "thresholds.json"
    if include_thresholds:
        thresholds_path.write_text(
            json.dumps(
                {
                    "f1": {"threshold": FROZEN_THRESHOLD, "rule": "protocol"},
                    "fpr10": {"threshold": 0.6, "rule": "protocol"},
                }
            ),
            encoding="utf-8",
        )

    scores, preds = _synthetic_frames()
    scores_path = tmp_path / "scores" / "example_scores.parquet"
    scores_path.parent.mkdir(parents=True)
    scores.to_parquet(scores_path, index=False)
    preds_path = test_dir / "predictions.parquet"
    preds.to_parquet(preds_path, index=False)

    config = LinkConfig(
        backend="nllb",
        example_scores_parquet=scores_path,
        predictions_parquet=preds_path,
        output_dir=tmp_path / "link",
        protocol_thresholds_json=thresholds_path,
        threshold_criterion="f1",
        expected_protocol_signature=PROTOCOL_SIGNATURE,
        quality_signal="nli_abs_delta_max",
        quality_direction="lower_is_better",
        bootstrap_samples=50,
    )
    return config, scores_path, preds_path


def test_link_end_to_end_writes_artifacts_with_frozen_threshold(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    result = link_quality_to_detection(config, write=True)

    assert result.threshold == pytest.approx(FROZEN_THRESHOLD)
    assert result.summary["n_examples"] == 64
    assert result.summary["threshold"] == pytest.approx(FROZEN_THRESHOLD)
    assert result.summary["threshold_rule"] == "protocol"
    for column in ("abs_error", "log_loss", "correct", "pred_at_threshold"):
        assert column in result.joined.columns

    run_dir = config.run_dir
    assert (run_dir / "linked_examples.parquet").is_file()
    assert (run_dir / "stratified_metrics.csv").is_file()
    assert (run_dir / "filtered_curve.csv").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "link_summary.json").is_file()


def test_stage2_never_reselects_threshold(tmp_path, monkeypatch):
    config, *_ = _write_inputs(tmp_path)

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("select_threshold não pode ser chamado no Stage 2")

    monkeypatch.setattr("ragtruth_transfer.metrics.select_threshold", _forbidden)
    assert not hasattr(linkage, "select_threshold")

    result = link_quality_to_detection(config, write=False)
    assert result.threshold == pytest.approx(FROZEN_THRESHOLD)


def test_protocol_threshold_is_read_from_artifact(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    threshold, record = load_protocol_threshold(config)
    assert threshold == pytest.approx(FROZEN_THRESHOLD)
    assert record["rule"] == "protocol"


def test_missing_threshold_criterion_raises(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    partial = tmp_path / "partial_thresholds.json"
    partial.write_text(json.dumps({"f1": {"threshold": FROZEN_THRESHOLD}}), encoding="utf-8")
    bad = replace(config, threshold_criterion="fpr10", protocol_thresholds_json=partial)
    with pytest.raises(ValueError):
        load_protocol_threshold(bad)


def test_stratified_auprc_increases_with_quality(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    result = link_quality_to_detection(config, write=False)
    strat = result.stratified.sort_values("quality_bucket")
    assert len(strat) == 4
    auprc = strat["AUPRC"].to_numpy()
    assert auprc[0] < auprc[-1]
    assert (strat["AUPRC_ci_low"] <= strat["AUPRC"] + 1e-9).all()


def test_join_fails_when_prediction_uncovered(tmp_path):
    config, scores_path, _ = _write_inputs(tmp_path)
    scores = pd.read_parquet(scores_path)
    scores.iloc[:-1].to_parquet(scores_path, index=False)
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_join_fails_on_label_mismatch(tmp_path):
    config, _, preds_path = _write_inputs(tmp_path)
    preds = pd.read_parquet(preds_path)
    preds.loc[0, "label"] = not bool(preds.loc[0, "label"])
    preds.to_parquet(preds_path, index=False)
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_join_fails_on_source_id_mismatch(tmp_path):
    config, _, preds_path = _write_inputs(tmp_path)
    preds = pd.read_parquet(preds_path)
    preds.loc[0, "source_id"] = "different-source"
    preds.to_parquet(preds_path, index=False)
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_join_fails_when_predictions_missing_source_id(tmp_path):
    config, _, preds_path = _write_inputs(tmp_path)
    preds = pd.read_parquet(preds_path).drop(columns=["source_id"])
    preds.to_parquet(preds_path, index=False)
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_join_fails_when_predictions_missing_label(tmp_path):
    config, _, preds_path = _write_inputs(tmp_path)
    preds = pd.read_parquet(preds_path).drop(columns=["label"])
    preds.to_parquet(preds_path, index=False)
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_backend_uses_confirmatory_metadata(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    protocol_dir = config.predictions_parquet.parents[2]
    (protocol_dir / "frozen_protocol.json").write_text(
        json.dumps(
            {
                "signature": PROTOCOL_SIGNATURE,
                "payload": {"protocol_name": "ragtruth_pt_nllb_confirmatory"},
            }
        ),
        encoding="utf-8",
    )
    result = link_quality_to_detection(config, write=False)
    assert result.summary["n_examples"] == 64


def test_backend_metadata_signature_mismatch_raises(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    protocol_dir = config.predictions_parquet.parents[2]
    (protocol_dir / "frozen_protocol.json").write_text(
        json.dumps(
            {
                "signature": "ffffffffffffffff",
                "payload": {"protocol_name": "ragtruth_pt_nllb_confirmatory"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_backend_metadata_protocol_mismatch_raises(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    protocol_dir = config.predictions_parquet.parents[2]
    (protocol_dir / "frozen_protocol.json").write_text(
        json.dumps(
            {
                "signature": PROTOCOL_SIGNATURE,
                "payload": {"protocol_name": "ragtruth_pt_madlad_confirmatory"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(config, write=False)


def test_backend_signature_validation(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    bad = LinkConfig(
        backend="nllb",
        example_scores_parquet=config.example_scores_parquet,
        predictions_parquet=config.predictions_parquet,
        output_dir=tmp_path / "link2",
        protocol_thresholds_json=config.protocol_thresholds_json,
        expected_protocol_signature="deadbeefdeadbeef",
    )
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(bad, write=False)


def test_missing_quality_signal_raises(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    bad = LinkConfig(
        backend="nllb",
        example_scores_parquet=config.example_scores_parquet,
        predictions_parquet=config.predictions_parquet,
        output_dir=tmp_path / "link3",
        protocol_thresholds_json=config.protocol_thresholds_json,
        expected_protocol_signature=PROTOCOL_SIGNATURE,
        quality_signal="does_not_exist",
    )
    with pytest.raises(LinkValidationError):
        link_quality_to_detection(bad, write=False)


def test_sample_limit_keeps_frozen_threshold(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    sampled = LinkConfig(
        backend=config.backend,
        example_scores_parquet=config.example_scores_parquet,
        predictions_parquet=config.predictions_parquet,
        output_dir=tmp_path / "link4",
        protocol_thresholds_json=config.protocol_thresholds_json,
        expected_protocol_signature=PROTOCOL_SIGNATURE,
        quality_signal="nli_abs_delta_max",
        quality_direction="lower_is_better",
        sample_limit=16,
        bootstrap_samples=20,
    )
    result = link_quality_to_detection(sampled, write=False)
    assert result.summary["n_examples"] == 16
    assert result.threshold == pytest.approx(FROZEN_THRESHOLD)


def test_add_error_columns_math(tmp_path):
    config, *_ = _write_inputs(tmp_path)
    frame = pd.DataFrame([{"label": 1, "score": 0.9}, {"label": 0, "score": 0.2}])
    out = add_error_columns(frame, config, threshold=0.5)
    assert out["pred_at_threshold"].tolist() == [1, 0]
    assert out["correct"].tolist() == [1, 1]
    assert out["abs_error"].iloc[0] == pytest.approx(0.1, abs=1e-6)
