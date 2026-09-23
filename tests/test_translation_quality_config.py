from __future__ import annotations

from pathlib import Path

import pytest

from ragtruth_transfer.translation_quality.config import (
    ALL_METRICS,
    AlignmentConfig,
    LinkConfig,
    ScoringConfig,
    alignment_from_mapping,
    link_from_mapping,
    resolve_device,
    scoring_from_mapping,
)


def _alignment(**kwargs) -> AlignmentConfig:
    base = dict(
        backend="nllb",
        en_parquet=Path("/data/en.parquet"),
        pt_parquet=Path("/data/pt.parquet"),
        output_dir=Path("/out"),
    )
    base.update(kwargs)
    return AlignmentConfig(**base)


def test_all_metrics_are_exactly_the_scoped_three():
    assert ALL_METRICS == ("heuristics", "cometkiwi", "nli_consistency")


def test_scoring_rejects_out_of_scope_metrics(tmp_path):
    for forbidden in ("labse", "backtranslation", "llm_judge", "bertscore", "chrf"):
        with pytest.raises(ValueError):
            ScoringConfig(
                backend="nllb",
                aligned_parquet=tmp_path / "a.parquet",
                output_dir=tmp_path / "out",
                metrics=(forbidden,),
            )


def test_scoring_has_no_seed_attribute(tmp_path):
    config = ScoringConfig(
        backend="nllb", aligned_parquet=tmp_path / "a.parquet", output_dir=tmp_path / "out"
    )
    assert not hasattr(config, "sample_seed")
    assert "seed" not in config.recipe()


def test_alignment_signature_is_machine_independent():
    a = _alignment(output_dir=Path("/laptop/out"), en_parquet=Path("/laptop/en.parquet"))
    b = _alignment(output_dir=Path("/cluster/out"), en_parquet=Path("/cluster/en.parquet"))
    assert a.signature == b.signature
    assert a.run_dir == Path("/laptop/out") / a.signature


def test_scoring_signature_ignores_device_and_batch(tmp_path):
    common = dict(
        backend="nllb", aligned_parquet=tmp_path / "a.parquet", output_dir=tmp_path / "out"
    )
    a = ScoringConfig(device="cpu", batch_size=8, **common)
    b = ScoringConfig(device="cuda", batch_size=64, **common)
    assert a.signature == b.signature


def test_scoring_signature_changes_with_metrics(tmp_path):
    common = dict(
        backend="nllb", aligned_parquet=tmp_path / "a.parquet", output_dir=tmp_path / "out"
    )
    a = ScoringConfig(metrics=("heuristics",), **common)
    b = ScoringConfig(
        metrics=("heuristics", "nli_consistency"),
        nli_model_revision="rev-nli",
        **common,
    )
    assert a.signature != b.signature


def test_scoring_requires_revision_when_metric_selected(tmp_path):
    common = dict(
        backend="nllb", aligned_parquet=tmp_path / "a.parquet", output_dir=tmp_path / "out"
    )
    with pytest.raises(ValueError):
        ScoringConfig(metrics=("cometkiwi",), **common)
    with pytest.raises(ValueError):
        ScoringConfig(metrics=("nli_consistency",), **common)
    with pytest.raises(ValueError):
        ScoringConfig(metrics=("heuristics",), detector_truncation=True, **common)


def test_scoring_signature_includes_revisions(tmp_path):
    common = dict(
        backend="nllb",
        aligned_parquet=tmp_path / "a.parquet",
        output_dir=tmp_path / "out",
        metrics=("heuristics", "cometkiwi", "nli_consistency"),
    )
    a = ScoringConfig(
        cometkiwi_model_revision="rev-comet-a",
        nli_model_revision="rev-nli",
        **common,
    )
    b = ScoringConfig(
        cometkiwi_model_revision="rev-comet-b",
        nli_model_revision="rev-nli",
        **common,
    )
    assert a.recipe()["cometkiwi_model_revision"] == "rev-comet-a"
    assert a.signature != b.signature


def test_scoring_entail_threshold_bounds(tmp_path):
    with pytest.raises(ValueError):
        ScoringConfig(
            backend="nllb",
            aligned_parquet=tmp_path / "a.parquet",
            output_dir=tmp_path / "out",
            entail_threshold=1.5,
        )


def test_link_validates_criterion_direction_buckets(tmp_path):
    base = dict(
        backend="nllb",
        example_scores_parquet=tmp_path / "s.parquet",
        predictions_parquet=tmp_path / "p.parquet",
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError):
        LinkConfig(threshold_criterion="nope", **base)
    with pytest.raises(ValueError):
        LinkConfig(quality_direction="sideways", **base)
    with pytest.raises(ValueError):
        LinkConfig(num_quality_buckets=1, **base)


def test_link_default_threshold_path_derives_from_predictions(tmp_path):
    config = LinkConfig(
        backend="nllb",
        example_scores_parquet=tmp_path / "s.parquet",
        predictions_parquet=tmp_path / "preds" / "predictions.parquet",
        output_dir=tmp_path / "out",
    )
    assert config.thresholds_json == tmp_path / "preds" / "thresholds_applied.json"


def test_full_yaml_mapping_round_trip(tmp_path):
    raw = {
        "backend": "nllb",
        "alignment": {
            "en_parquet": "en.parquet",
            "pt_parquet": "pt.parquet",
            "expected_rows": 34604,
        },
        "scoring": {
            "metrics": ["heuristics", "cometkiwi", "nli_consistency"],
            "sample_split": "test",
            "sample_limit": 128,
            "cometkiwi_model_revision": "rev-comet",
            "nli_model_revision": "rev-nli",
        },
        "link": {
            "predictions_parquet": "preds.parquet",
            "expected_protocol_signature": "70bb1cce59b8c824",
            "quality_signal": "nli_abs_delta_max",
        },
    }
    base = tmp_path

    alignment = alignment_from_mapping(raw, base)
    assert alignment.en_parquet == (base / "en.parquet").resolve()

    scoring = scoring_from_mapping(raw, base)
    assert scoring.metrics == ("heuristics", "cometkiwi", "nli_consistency")
    assert scoring.sample_limit == 128
    assert scoring.cometkiwi_model_revision == "rev-comet"
    assert scoring.nli_model_revision == "rev-nli"
    assert scoring.aligned_parquet == alignment.run_dir / "aligned.parquet"

    link = link_from_mapping(raw, base)
    assert link.predictions_parquet == (base / "preds.parquet").resolve()
    assert link.example_scores_parquet == scoring.run_dir / "example_scores.parquet"
    assert link.expected_protocol_signature == "70bb1cce59b8c824"


def test_link_requires_predictions(tmp_path):
    with pytest.raises(ValueError):
        link_from_mapping({"backend": "nllb", "link": {}}, tmp_path)


def test_resolve_device_cpu_and_invalid():
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        resolve_device("gpu")
