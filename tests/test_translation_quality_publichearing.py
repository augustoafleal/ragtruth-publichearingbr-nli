from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ragtruth_transfer.translation_quality.alignment import (
    AlignmentIntegrityError,
    align_translation_quality,
)
from ragtruth_transfer.translation_quality.config import AlignmentConfig, ScoringConfig
from ragtruth_transfer.translation_quality.scoring import ScorerBundle, score_translation_quality


def _record(*, record_id: int = 7, label: bool = False, claim: str = "A opinião em português", chunks=None) -> dict:
    return {
        "id": record_id,
        "metadados_extraidos": {
            "assunto": "Assunto",
            "envolvidos": [
                {
                    "nome": "Pessoa",
                    "cargo": "Cargo",
                    "opinioes": [
                        {
                            "opiniao": claim,
                            "chunks_proximos": chunks or ["e1", "e2", "e3", "e4"],
                            "verificacao_alucinacao": {"verificacao_manual": label},
                        }
                    ],
                }
            ],
        },
    }


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def _config(tmp_path: Path, source: Path, translated: Path, **kwargs) -> AlignmentConfig:
    return AlignmentConfig(
        backend="nllb",
        source_artifact=source,
        translated_artifact=translated,
        output_dir=tmp_path / "alignment",
        source_language="pt",
        target_language="en",
        source_format="jsonl",
        translated_format="jsonl",
        expected_rows=1,
        **kwargs,
    )


def test_publichearing_alignment_uses_strict_nested_fixture(tmp_path):
    source = tmp_path / "source.jsonl"
    translated = tmp_path / "translated.jsonl"
    _jsonl(source, [_record(label=True)])
    _jsonl(translated, [_record(label=True, claim="The opinion in English", chunks=["a", "b", "c", "d"])])

    result = align_translation_quality(_config(tmp_path, source, translated))

    assert result.counts.gate_ok
    assert result.frame.loc[0, "example_id"] == "7:0:0"
    assert result.frame.loc[0, "source_id"] == "7"
    assert result.frame.loc[0, "claim_source"] == "A opinião em português"
    assert result.frame.loc[0, "claim_target"] == "The opinion in English"
    manifest = json.loads((result.config.run_dir / "manifest.json").read_text())
    assert manifest["config"]["source_language"] == "pt"
    assert manifest["config"]["target_language"] == "en"


def test_publichearing_alignment_rejects_coverage_and_label_mismatch(tmp_path):
    source = tmp_path / "source.jsonl"
    translated = tmp_path / "translated.jsonl"
    _jsonl(source, [_record(label=True)])
    _jsonl(translated, [_record(label=False)])
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(_config(tmp_path, source, translated), validate_only=True)
    assert error.value.counts.label_mismatches == 1

    _jsonl(translated, [_record(label=True, chunks=["a", "b", "c", "d"]), _record(record_id=8, label=True)])
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(_config(tmp_path, source, translated), validate_only=True)
    assert error.value.counts.en_extra_rows == 1


def test_direction_is_part_of_alignment_and_scoring_signatures(tmp_path):
    source = tmp_path / "source.parquet"
    translated = tmp_path / "translated.parquet"
    a = AlignmentConfig("x", tmp_path / "out", source_artifact=source, translated_artifact=translated, source_language="pt", target_language="en")
    b = AlignmentConfig("x", tmp_path / "out", source_artifact=source, translated_artifact=translated, source_language="en", target_language="pt")
    assert a.signature != b.signature
    aligned = tmp_path / "aligned.parquet"
    pd.DataFrame({"example_id": ["e"], "source_id": ["s"], "split": ["all"], "label": [True], "claim_source": ["claim pt"], "claim_target": ["claim en"], "chunk_1_valid": [True], "chunk_1_source": ["evidence pt"], "chunk_1_target": ["evidence en"], "chunk_2_valid": [False], "chunk_3_valid": [False], "chunk_4_valid": [False]}).to_parquet(aligned, index=False)
    pt = ScoringConfig("x", aligned, tmp_path / "scores", source_language="pt", target_language="en")
    en = ScoringConfig("x", aligned, tmp_path / "scores", source_language="en", target_language="pt")
    assert pt.signature != en.signature


class _RecordingComet:
    def score_pairs(self, pairs):
        self.pairs = list(pairs)
        return [{"cometkiwi": 0.5} for _ in pairs]


class _RecordingNLI:
    def __init__(self):
        self.calls = []

    def entailment_probs(self, pairs):
        self.calls.append(list(pairs))
        return [0.2 if "pt" in premise else 0.8 for premise, _ in pairs]


class _Tokenizer:
    def __call__(self, premise, hypothesis, truncation=False, return_attention_mask=False):
        return {"input_ids": list(range(len(str(premise).split()) + len(str(hypothesis).split())))}


def test_publichearing_scoring_passes_pt_as_source_and_en_as_translation(tmp_path):
    aligned = tmp_path / "aligned.parquet"
    pd.DataFrame([{"example_id": "e", "source_id": "7", "split": "all", "label": True, "claim_source": "claim pt", "claim_target": "claim en", "chunk_1_valid": True, "chunk_1_source": "evidence pt", "chunk_1_target": "evidence en", "chunk_2_valid": False, "chunk_3_valid": False, "chunk_4_valid": False}]).to_parquet(aligned, index=False)
    config = ScoringConfig("nllb", aligned, tmp_path / "scores", source_language="pt", target_language="en")
    comet = _RecordingComet()
    nli = _RecordingNLI()
    result = score_translation_quality(config, scorers=ScorerBundle(cometkiwi=comet, nli=nli, detector_tokenizer=_Tokenizer()), write=False)

    assert comet.pairs[0] == ("claim pt", "claim en")
    assert nli.calls[0][0] == ("evidence pt", "claim pt")
    assert nli.calls[1][0] == ("evidence en", "claim en")
    row = result.segment_frame[result.segment_frame["role"] == "chunk"].iloc[0]
    assert row["nli_entail_source"] == pytest.approx(0.2)
    assert row["nli_entail_target"] == pytest.approx(0.8)
    assert row["nli_entail_delta"] == pytest.approx(0.6)
    assert "detector_truncation_introduced" in result.segment_frame
