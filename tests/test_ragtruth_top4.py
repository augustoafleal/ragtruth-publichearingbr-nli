from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ragtruth_transfer.ragtruth_top4_config import DatasetSettings, RetrieverSettings, Top4Config
from ragtruth_transfer.ragtruth_top4_embeddings import (
    EmbeddingCache,
    MockTextEncoder,
    WhitespaceTokenizer,
    chunk_passage,
    signature_for,
)
from ragtruth_transfer.ragtruth_top4_pipeline import (
    RawTop4Example,
    _score_examples,
    build_audit,
    load_qa_examples,
    parse_qa_passages_indexed,
    prepare_top4,
)
from ragtruth_transfer.ragtruth_top4_pipeline import _embedding_signature
from ragtruth_transfer.ragtruth_data import response_claims


def _config(tmp_path: Path, response: Path, source: Path) -> Top4Config:
    return Top4Config(
        run_name="test",
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        dataset=DatasetSettings(response, source, splits=("train", "test"), min_words=1),
        retriever=RetrieverSettings(
            model_id="mock-hash-encoder",
            revision="local-v1",
            tokenizer_model_id="mock-whitespace",
            tokenizer_revision="local-v1",
            encoder_backend="mock",
            embedding_dim=8,
            chunk_size_tokens=4,
            chunk_overlap_tokens=2,
            batch_size=2,
            device="cpu",
        ),
    )


def _raw_files(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source_info.jsonl"
    response = tmp_path / "response.jsonl"
    source.write_text(
        json.dumps({"source_id": "s1", "task_type": "QA", "source": "unit", "source_info": {"question": "what?", "passages": "passage 1: alpha beta gamma delta\n\npassage 2: epsilon zeta eta theta"}})
        + "\n"
        + json.dumps({"source_id": "s2", "task_type": "QA", "source": "unit", "source_info": {"question": "where?", "passages": "passage 1: one two three four"}})
        + "\n",
        encoding="utf-8",
    )
    response.write_text(
        json.dumps({"id": "r1", "source_id": "s1", "split": "train", "quality": "good", "model": "mock", "labels": [], "response": "alpha beta"})
        + "\n"
        + json.dumps({"id": "r2", "source_id": "s2", "split": "test", "quality": "good", "model": "mock", "labels": [{"start": 0, "end": 4, "text": "one ", "label_type": "Evident Conflict"}], "response": "one two"})
        + "\n",
        encoding="utf-8",
    )
    return response, source


def test_parse_passages_and_chunk_overlap_are_deterministic() -> None:
    assert parse_qa_passages_indexed("passage 1: a\n\npassage 3: c") == [(0, "a"), (2, "c")]
    tokenizer = WhitespaceTokenizer()
    chunks = chunk_passage("one two three four five six seven eight nine", 2, tokenizer, 4, 2, "s")
    assert [(x.token_start, x.token_end) for x in chunks] == [(0, 4), (2, 6), (4, 8), (6, 9)]
    assert chunks == chunk_passage("one two three four five six seven eight nine", 2, tokenizer, 4, 2, "s")


def test_empty_passages_are_excluded() -> None:
    chunks = chunk_passage("   ", 0, WhitespaceTokenizer(), 4, 0, "s")
    assert chunks == []


def test_top4_tie_breaking_padding_and_label_independence() -> None:
    chunks = chunk_passage("a b c d", 0, WhitespaceTokenizer(), 4, 0, "s")
    false_example = RawTop4Example("e0", "s", "r", "train", "claim", False, 0, 0, 5, "", "", "", None, [], [], [], chunks)
    true_example = replace(false_example, example_id="e1", label=True)
    vectors = {chunk.chunk_key: np.ones(4, dtype=np.float32) / 2 for chunk in chunks}
    claims = {"claim-key": np.ones(4, dtype=np.float32) / 2}
    import hashlib
    claims = {hashlib.sha256("claim".encode()).hexdigest(): np.ones(4, dtype=np.float32) / 2}
    rows_false = _score_examples([false_example], vectors, claims)
    rows_true = _score_examples([true_example], vectors, claims)
    assert rows_false[0]["evidence_mask"] == [True, False, False, False]
    assert rows_false[0]["chunk_1_score"] == rows_true[0]["chunk_1_score"]
    assert rows_false[0]["label"] is False and rows_true[0]["label"] is True


def test_top4_relevance_is_sorted_descending() -> None:
    chunks = chunk_passage("a b c d e f g h", 0, WhitespaceTokenizer(), 2, 0, "s")
    example = RawTop4Example("e0", "s", "r", "train", "claim", False, 0, 0, 5, "", "", "", None, [], [], [], chunks)
    vectors = {chunk.chunk_key: np.array([float(chunk.window_index + 1), 0.0], dtype=np.float32) for chunk in chunks}
    claim = np.array([1.0, 0.0], dtype=np.float32)
    import hashlib
    rows = _score_examples([example], vectors, {hashlib.sha256(b"claim").hexdigest(): claim})
    scores = [rows[0][f"chunk_{i}_score"] for i in range(1, 5)]
    assert scores == sorted(scores, reverse=True)


def test_cache_hit_invalidation_and_truncation(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, "claims", "sig")
    encoder = MockTextEncoder(4)
    keys = ["a", "b"]
    values = encoder.encode(["one", "two"], "query: ", 2)
    cache.save(keys, values)
    np.testing.assert_allclose(cache.load(keys, 4), values)
    assert cache.load(["a"], 4) is None
    cache.data_path.write_bytes(b"truncated")
    assert cache.load(keys, 4) is None


def test_cache_data_hash_detects_mutation(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, "chunks", "sig")
    values = MockTextEncoder(4).encode(["one"], "passage: ", 1)
    cache.save(["key"], values)
    data = bytearray(cache.data_path.read_bytes())
    data[-1] ^= 1
    cache.data_path.write_bytes(data)
    assert cache.load(["key"], 4) is None


def test_load_qa_preserves_official_splits_and_source_isolation(tmp_path: Path) -> None:
    response, source = _raw_files(tmp_path)
    config = _config(tmp_path, response, source)
    rows, rejected, stats = load_qa_examples(config)
    assert {row.split for row in rows} == {"train", "test"}
    assert {row.source_id for row in rows} == {"s1", "s2"}
    assert stats["rejected_count"] == len(rejected)
    assert all(chunk.chunk_key.startswith(f"{row.source_id}:") for row in rows for chunk in row.candidates)
    assert any(row.label is False for row in rows)
    assert any(row.label is True and row.span_texts for row in rows)


def test_claim_spans_keep_negative_sentences_and_merge_cross_sentence_spans() -> None:
    text = "Supported sentence. First unsupported sentence. Second unsupported sentence."
    start = text.index("First")
    response = {"response": text, "labels": [{"start": start, "end": len(text), "text": text[start:]}]}
    claims = response_claims(response, granularity="claim", min_words=1)
    assert [row["label"] for row in claims] == [False, True]
    assert claims[1]["claim"] == text[start:]
    assert claims[1]["labels"][0]["text"] == text[start:]


def test_transformer_config_requires_immutable_revisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="revision"):
        Top4Config(
            run_name="bad",
            output_root=tmp_path,
            cache_root=tmp_path,
            dataset=DatasetSettings(tmp_path / "r", tmp_path / "s"),
            retriever=RetrieverSettings(encoder_backend="transformers", revision=None, tokenizer_revision=None),
        )


def test_prepare_writes_parquet_manifest_audit_and_resume(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    response, source = _raw_files(tmp_path)
    config = _config(tmp_path, response, source)
    output = tmp_path / "output"
    manifest = prepare_top4(config, output_dir=output, limit=2)
    assert manifest["counts"]["examples"] == 2
    assert (output / "dataset.parquet").is_file()
    import pandas as pd
    frame = pd.read_parquet(output / "dataset.parquet")
    assert len(frame) == 2
    assert all(len(mask) == 4 for mask in frame["evidence_mask"])
    resumed = prepare_top4(config, output_dir=output, limit=2, resume=True)
    assert resumed["signature"] == manifest["signature"]
    assert (output / "_parts" / "manifest.json").is_file()
    broken = bytearray((output / "dataset.parquet").read_bytes())
    broken[-1] ^= 1
    (output / "dataset.parquet").write_bytes(broken)
    repaired = prepare_top4(config, output_dir=output, limit=2, resume=True)
    assert repaired["artifacts"]["dataset.parquet"] != ""
    assert pd.read_parquet(output / "dataset.parquet").shape[0] == 2


def test_audit_rejects_no_integrity_errors_and_has_four_slots(tmp_path: Path) -> None:
    response, source = _raw_files(tmp_path)
    config = _config(tmp_path, response, source)
    rows, rejected, _ = load_qa_examples(config)
    import hashlib
    embeddings = {chunk.chunk_key: np.ones(8, dtype=np.float32) / np.sqrt(8) for row in rows for chunk in row.candidates}
    claims = {hashlib.sha256(row.claim.encode()).hexdigest(): np.ones(8, dtype=np.float32) / np.sqrt(8) for row in rows}
    scored = _score_examples(rows, embeddings, claims)
    audit = build_audit(scored, rejected, config)
    assert audit["integrity"]["errors"] == []
    assert audit["integrity"]["exactly_four_slots"] is True


def test_cli_estimate_does_not_load_encoder(tmp_path: Path) -> None:
    response, source = _raw_files(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join([
            "run_name: cli-test", f"output_root: {tmp_path / 'out'}", f"cache_root: {tmp_path / 'cache'}", "dataset:",
            f"  response_path: {response}", f"  source_path: {source}", "  task: QA", "  splits: [train, test]",
            "retriever:", "  encoder_backend: mock", "  model_id: mock", "  tokenizer_model_id: mock", "  embedding_dim: 4",
        ]),
        encoding="utf-8",
    )
    result = subprocess.run([sys.executable, "scripts/prepare_ragtruth_top4.py", "--config", str(config_path), "--estimate", "--limit", "1"], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["encoder_loaded"] is False


def test_signature_stable_and_changes_with_configuration() -> None:
    value = {"schema": 1, "chunk_size": 256}
    assert signature_for(value) == signature_for({"chunk_size": 256, "schema": 1})
    assert signature_for(value) != signature_for({"schema": 1, "chunk_size": 128})


def test_embedding_signature_changes_with_revision_and_chunking(tmp_path: Path) -> None:
    response, source = _raw_files(tmp_path)
    config = _config(tmp_path, response, source)
    dataset = {"response_sha256": "r", "source_sha256": "s"}
    original = _embedding_signature(config, dataset, "chunks")
    changed_chunk = replace(config, retriever=replace(config.retriever, chunk_size_tokens=8))
    changed_revision = replace(config, retriever=replace(config.retriever, revision="other"))
    assert original != _embedding_signature(changed_chunk, dataset, "chunks")
    assert original != _embedding_signature(changed_revision, dataset, "chunks")
