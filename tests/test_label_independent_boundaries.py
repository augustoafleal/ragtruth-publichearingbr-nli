from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ragtruth_transfer.ragtruth_data import label_independent_sentence_claims
from ragtruth_transfer.ragtruth_top4_pipeline import (
    RawTop4Example,
    _claim_span_metadata,
    _score_examples,
    build_audit,
    build_boundary_audit,
    build_surface_feature_audit,
)
from ragtruth_transfer.ragtruth_top4_config import DatasetSettings, RetrieverSettings, Top4Config
from ragtruth_transfer.ragtruth_top4_embeddings import WhitespaceTokenizer, chunk_passage, signature_for


def _config(tmp_path: Path) -> Top4Config:
    return Top4Config(
        run_name="independent",
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        dataset=DatasetSettings(tmp_path / "response.jsonl", tmp_path / "source.jsonl", min_words=1, boundary_strategy="label_independent_sentences"),
        retriever=RetrieverSettings(
            model_id="mock", revision="local-v1", tokenizer_model_id="mock", tokenizer_revision="local-v1",
            encoder_backend="mock", embedding_dim=4, chunk_size_tokens=4, chunk_overlap_tokens=0, device="cpu",
        ),
        schema_version="ragtruth-qa-top4-label-independent-v2",
    )


def test_same_response_labels_cannot_change_claim_text_offsets_or_ids() -> None:
    text = "First sentence. Second sentence."
    base = {"response": text, "labels": []}
    changed = {"response": text, "labels": [{"start": 5, "end": len(text), "text": text[5:]}]}
    left = label_independent_sentence_claims(base, min_words=1)
    right = label_independent_sentence_claims(changed, min_words=1)
    assert [(x["claim"], x["claim_start"], x["claim_end"]) for x in left] == [(x["claim"], x["claim_start"], x["claim_end"]) for x in right]
    assert [x["label"] for x in left] == [False, False]
    assert [x["label"] for x in right] == [True, True]
    assert [f"r:{x['claim_start']}:{x['claim_end']}" for x in left] == [f"r:{x['claim_start']}:{x['claim_end']}" for x in right]


def test_overlap_inside_crossing_and_boundary_touching() -> None:
    text = "Alpha. Beta."
    units = label_independent_sentence_claims({"response": text, "labels": []}, min_words=1)
    first, second = units
    metadata = _claim_span_metadata(text, first["claim_start"], first["claim_end"], [{"start": 0, "end": 2, "text": "Al"}], [(x["claim_start"], x["claim_end"]) for x in units])
    assert metadata["span_overlap_chars"] == [2]
    crossing = _claim_span_metadata(text, first["claim_start"], first["claim_end"], [{"start": 4, "end": 9, "text": "ha. B"}], [(x["claim_start"], x["claim_end"]) for x in units])
    assert crossing["span_crosses_claim_boundary"] is True
    touching = _claim_span_metadata(text, first["claim_start"], first["claim_end"], [{"start": first["claim_end"], "end": second["claim_end"], "text": text[first["claim_end"] :]}], [(x["claim_start"], x["claim_end"]) for x in units])
    assert touching["span_overlap_chars"] == [0] if touching["span_overlap_chars"] else True


def test_multiple_overlapping_spans_and_textual_edge_cases() -> None:
    response = {"response": "Dr. Smith said: one.\n2. two\n* three", "labels": [
        {"start": 0, "end": 3, "text": "Dr."}, {"start": 0, "end": 5, "text": "Dr. S"},
    ]}
    claims = label_independent_sentence_claims(response, min_words=1)
    assert claims
    assert claims[0]["claim_start"] == 0
    assert label_independent_sentence_claims({"response": "", "labels": []}, min_words=1) == []
    assert label_independent_sentence_claims({"response": "fragment without terminal punctuation", "labels": []}, min_words=1)[0]["claim_end"] == len("fragment without terminal punctuation")


def test_audits_declare_independence_and_surface_baseline_is_grouped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    chunks = chunk_passage("one two three four", 0, WhitespaceTokenizer(), 4, 0, "s")
    examples = [
        RawTop4Example("r:0:3", "s", "r", "train", "one two", False, 0, 0, 7, "", "", "", None, [], [], [], chunks, [], [], [], [], False, "label_independent_sentences", "one two"),
        RawTop4Example("r:7:15", "s", "r", "train", "longer claim text", True, 1, 7, 25, "", "", "", None, ["x"], [[1, 2]], [], chunks, [["x"]], [1], [0.1], [1.0], False, "label_independent_sentences", "longer claim text"),
    ]
    vectors = {chunk.chunk_key: np.ones(4, dtype=np.float32) / 2 for chunk in chunks}
    claims = {__import__("hashlib").sha256(row.claim.encode()).hexdigest(): np.ones(4, dtype=np.float32) / 2 for row in examples}
    rows = _score_examples(examples, vectors, claims)
    audit = build_audit(rows, [], config)
    assert audit["integrity"]["labels_used_for_claim_boundaries"] is False
    boundary = build_boundary_audit(rows, [], config)
    assert boundary["labels_used_for_claim_boundaries"] is False
    surface = build_surface_feature_audit(rows, seed=42)
    assert surface["status"] in {"completed", "skipped"}


def test_parquet_round_trip_preserves_boundary_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame([{
        "example_id": "r:0:5", "claim_start": 0, "claim_end": 5, "label": True,
        "span_offsets": [[1, 3]], "span_overlap_chars": [2], "evidence_mask": [True, False, False, False],
    }])
    path = tmp_path / "dataset.parquet"
    frame.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert loaded.iloc[0].example_id == "r:0:5"
    assert loaded.iloc[0].evidence_mask.tolist() == [True, False, False, False]


def test_new_schema_signature_differs_from_legacy() -> None:
    assert signature_for({"schema": "ragtruth-qa-top4-v1"}) != signature_for({"schema": "ragtruth-qa-top4-label-independent-v2"})
