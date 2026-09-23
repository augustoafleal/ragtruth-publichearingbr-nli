from __future__ import annotations

import pandas as pd
import pytest

from ragtruth_transfer.translation_quality.config import ScoringConfig
from ragtruth_transfer.translation_quality.metrics import (
    CometKiwiScorer,
    HeuristicScorer,
    NLIEntailmentScorer,
)
from ragtruth_transfer.translation_quality.scoring import (
    ScorerBundle,
    aggregate_examples,
    build_scorers,
    score_translation_quality,
)

FORBIDDEN_SUBSTRINGS = ("labse", "backtranslation", "chrf", "llm_judge", "bertscore", "comet_")


class FakeCometKiwi:
    name = "cometkiwi"

    def score_pairs(self, pairs):
        return [{"cometkiwi": 0.2 if "ruim" in str(mt) else 0.9} for _src, mt in pairs]


class FakeNLI:
    name = "nli_consistency"

    def entailment_probs(self, pairs):
        return [0.1 if "flip" in str(premise) else 0.9 for premise, _hyp in pairs]


class FakeTokenizer:
    def __call__(self, premise, hypothesis, truncation=False, return_attention_mask=False):
        n = len(str(premise).split()) + len(str(hypothesis).split())
        return {"input_ids": list(range(n))}


def _aligned_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "example_id": "ex-good",
                "source_id": "s1",
                "split": "test",
                "label": True,
                "claim_en": "the cat sat",
                "claim_pt": "o gato sentou",
                "chunk_1_valid": True,
                "chunk_1_en": "a cat was sitting on the mat",
                "chunk_1_pt": "um gato estava sentado no tapete agora mesmo",
                "chunk_2_valid": True,
                "chunk_2_en": "the animal rested",
                "chunk_2_pt": "o animal descansou",
                "chunk_3_valid": False,
                "chunk_3_en": "",
                "chunk_3_pt": "",
                "chunk_4_valid": False,
                "chunk_4_en": "",
                "chunk_4_pt": "",
            },
            {
                "example_id": "ex-bad",
                "source_id": "s2",
                "split": "test",
                "label": False,
                "claim_en": "the sky is blue",
                "claim_pt": "o ceu e ruim",
                "chunk_1_valid": True,
                "chunk_1_en": "flip the meaning here",
                "chunk_1_pt": "inverte o sentido aqui",
                "chunk_2_valid": True,
                "chunk_2_en": "short source",
                "chunk_2_pt": "x" * 500,
                "chunk_3_valid": False,
                "chunk_3_en": "",
                "chunk_3_pt": "",
                "chunk_4_valid": False,
                "chunk_4_en": "",
                "chunk_4_pt": "",
            },
        ]
    )


def _config(tmp_path, **kwargs) -> ScoringConfig:
    parquet = tmp_path / "aligned.parquet"
    _aligned_frame().to_parquet(parquet, index=False)
    kwargs.setdefault("detector_max_length", 10)
    kwargs.setdefault("metrics", ("heuristics",))
    return ScoringConfig(
        backend="test",
        aligned_parquet=parquet,
        output_dir=tmp_path / "scores",
        **kwargs,
    )


def test_heuristics_scoring_and_no_forbidden_columns(tmp_path):
    config = _config(tmp_path)
    result = score_translation_quality(
        config, scorers=ScorerBundle(heuristics=HeuristicScorer()), write=False
    )
    columns = list(result.segment_frame.columns)
    assert "heur_length_ratio" in columns
    assert "heur_exclusion_candidate" in columns
    for token in FORBIDDEN_SUBSTRINGS:
        assert not any(token in column for column in columns)

    ex = result.example_frame.set_index("example_id")
    assert bool(ex.loc["ex-bad", "any_high_ratio"]) is True
    assert bool(ex.loc["ex-bad", "any_exclusion_candidate"]) is True


def test_cometkiwi_fake_scoring_and_aggregation(tmp_path):
    config = _config(tmp_path)
    result = score_translation_quality(
        config, scorers=ScorerBundle(cometkiwi=FakeCometKiwi()), write=False
    )
    seg = result.segment_frame
    assert "cometkiwi" in seg.columns
    ex = result.example_frame.set_index("example_id")
    assert ex.loc["ex-bad", "cometkiwi_claim"] == pytest.approx(0.2)
    assert ex.loc["ex-good", "cometkiwi_chunk_min"] == pytest.approx(0.9)


def test_nli_consistency_signals(tmp_path):
    config = _config(tmp_path)
    result = score_translation_quality(
        config, scorers=ScorerBundle(nli=FakeNLI()), write=False
    )
    seg = result.segment_frame
    flip_row = seg[(seg["example_id"] == "ex-bad") & (seg["slot"] == 1)].iloc[0]
    assert flip_row["nli_entail_en"] == pytest.approx(0.1)
    assert flip_row["nli_entail_pt"] == pytest.approx(0.9)
    assert flip_row["nli_entail_abs_delta"] == pytest.approx(0.8)
    assert bool(flip_row["nli_label_agree"]) is False

    ex = result.example_frame.set_index("example_id")
    assert ex.loc["ex-bad", "nli_abs_delta_max"] == pytest.approx(0.8)
    assert bool(ex.loc["ex-bad", "nli_any_flip"]) is True
    assert bool(ex.loc["ex-good", "nli_any_flip"]) is False


def test_detector_truncation_is_diagnostic(tmp_path):
    config = _config(tmp_path, detector_max_length=10)
    result = score_translation_quality(
        config,
        scorers=ScorerBundle(nli=FakeNLI(), detector_tokenizer=FakeTokenizer()),
        write=False,
    )
    seg = result.segment_frame
    good = seg[(seg["example_id"] == "ex-good") & (seg["slot"] == 1)].iloc[0]
    assert good["detector_pair_tokens_en"] == 10
    assert good["detector_pair_tokens_pt"] == 11
    assert bool(good["detector_truncated_en"]) is False
    assert bool(good["detector_truncation_introduced"]) is True

    ex = result.example_frame.set_index("example_id")
    assert bool(ex.loc["ex-good", "any_truncation_introduced"]) is True


def test_aggregation_preserves_signals_without_composite(tmp_path):
    config = _config(tmp_path)
    bundle = ScorerBundle(
        heuristics=HeuristicScorer(),
        cometkiwi=FakeCometKiwi(),
        nli=FakeNLI(),
        detector_tokenizer=FakeTokenizer(),
    )
    result = score_translation_quality(config, scorers=bundle, write=True)
    columns = list(result.example_frame.columns)
    assert "example_quality" not in columns
    assert "task_risk" not in columns
    for expected in (
        "any_exclusion_candidate",
        "cometkiwi_claim",
        "nli_abs_delta_max",
        "nli_any_flip",
        "any_truncation_introduced",
    ):
        assert expected in columns

    assert (config.run_dir / "segment_scores.parquet").is_file()
    assert (config.run_dir / "example_scores.parquet").is_file()
    assert (config.run_dir / "quality_summary.json").is_file()


def test_aggregate_handles_missing_metrics():
    seg = pd.DataFrame(
        [
            {"example_id": "e1", "source_id": "s", "split": "test", "label": True, "role": "claim", "slot": 0},
            {"example_id": "e1", "source_id": "s", "split": "test", "label": True, "role": "chunk", "slot": 1},
        ]
    )
    out = aggregate_examples(seg)
    assert len(out) == 1
    assert out.iloc[0]["num_valid_chunks"] == 1


def test_truncation_works_without_nli(tmp_path):
    config = _config(tmp_path, detector_max_length=10)
    result = score_translation_quality(
        config,
        scorers=ScorerBundle(heuristics=HeuristicScorer(), detector_tokenizer=FakeTokenizer()),
        write=False,
    )
    seg = result.segment_frame
    assert "detector_truncation_introduced" in seg.columns
    assert "nli_entail_abs_delta" not in seg.columns
    ex = result.example_frame.set_index("example_id")
    assert bool(ex.loc["ex-good", "any_truncation_introduced"]) is True


def test_build_scorers_loads_tokenizer_without_nli(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        metrics=("heuristics",),
        detector_truncation=True,
        detector_tokenizer_revision="rev-tok",
    )
    monkeypatch.setattr(
        "ragtruth_transfer.translation_quality.scoring._load_detector_tokenizer",
        lambda _config: FakeTokenizer(),
    )
    bundle = build_scorers(config)
    assert bundle.nli is None
    assert bundle.detector_tokenizer is not None


def test_build_scorers_skips_tokenizer_when_truncation_disabled(tmp_path, monkeypatch):
    config = _config(tmp_path, metrics=("heuristics",), detector_truncation=False)

    def _fail(_config):  # pragma: no cover - must not run
        raise AssertionError("tokenizer não deve ser carregado com truncation desabilitado")

    monkeypatch.setattr(
        "ragtruth_transfer.translation_quality.scoring._load_detector_tokenizer", _fail
    )
    bundle = build_scorers(config)
    assert bundle.detector_tokenizer is None


def test_manifest_records_model_revisions(tmp_path):
    import json

    config = _config(
        tmp_path,
        metrics=("nli_consistency",),
        nli_model_revision="rev-nli",
        detector_max_length=10,
    )
    score_translation_quality(config, scorers=ScorerBundle(nli=FakeNLI()), write=True)
    manifest = json.loads((config.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["nli_model_revision"] == "rev-nli"


def test_nli_loader_uses_revision(monkeypatch):
    captured: dict = {}

    class FakeConfig:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    import transformers

    def fake_tokenizer(model_id, **kwargs):
        captured["tokenizer"] = (model_id, kwargs.get("revision"))
        return object()

    def fake_model(model_id, **kwargs):
        captured["model"] = (model_id, kwargs.get("revision"))
        return FakeModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer))
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", staticmethod(fake_model)
    )
    NLIEntailmentScorer.load("some-model", "rev-xyz", "cpu", 4, 128)
    assert captured["tokenizer"] == ("some-model", "rev-xyz")
    assert captured["model"] == ("some-model", "rev-xyz")


def test_cometkiwi_loader_uses_revision_and_nested_checkpoint(monkeypatch, tmp_path):
    import sys
    import types

    captured: dict = {}

    class FakeCometModel:
        def predict(self, *args, **kwargs):  # pragma: no cover - not called here
            return {"scores": []}

    fake_comet = types.ModuleType("comet")
    fake_comet.load_from_checkpoint = lambda path: captured.setdefault("path", path) or FakeCometModel()
    monkeypatch.setitem(sys.modules, "comet", fake_comet)

    import huggingface_hub

    snapshot_dir = tmp_path / "fake-checkpoint"
    checkpoint_path = snapshot_dir / "checkpoints" / "model.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fake checkpoint")

    def fake_snapshot(**kwargs):
        captured.update(kwargs)
        return str(snapshot_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    CometKiwiScorer.load("Unbabel/wmt22-cometkiwi-da", "rev-comet", "cpu", 8)
    assert captured["repo_id"] == "Unbabel/wmt22-cometkiwi-da"
    assert captured["revision"] == "rev-comet"
    assert captured["path"] == str(checkpoint_path)
