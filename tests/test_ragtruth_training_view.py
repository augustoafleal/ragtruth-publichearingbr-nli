from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ragtruth_transfer.io_utils import sha256_file
from ragtruth_transfer.ragtruth_training_view import (
    PARENT_SCHEMA,
    SUPPORTED_POLICY,
    TRAINING_VIEW_SCHEMA,
    build_training_view,
    duplicate_group_id,
    input_differences,
)


def _row(source: str, claim: str, response: str, label: bool, start: int = 0, model: str = "m", split: str = "train") -> dict:
    return {
        "example_id": f"{response}:{start}:{start + len(claim)}",
        "source_id": source,
        "response_id": response,
        "split": split,
        "task": "QA",
        "claim": claim,
        "label": label,
        "claim_index": 0,
        "claim_start": start,
        "claim_end": start + len(claim),
        "question": "q",
        "source_name": "unit",
        "generator_model": model,
        "model": model,
        "temperature": 0.7,
        "label_types": [],
        "span_texts": [],
        "span_offsets": [],
        "span_label_types": [],
        "span_overlap_chars": [],
        "span_overlap_fraction_claim": [],
        "span_overlap_fraction_spans": [],
        "span_crosses_claim_boundary": False,
        "claim_boundary_strategy": "label_independent_sentences",
        "label_span_count": 0,
        "num_candidate_chunks": 1,
        "evidence_mask": [True, False, False, False],
        "chunk_1": "evidence",
        "chunk_1_score": 0.9,
        "chunk_1_source_index": 0,
        "chunk_1_window_index": 0,
        "chunk_1_token_start": 0,
        "chunk_1_token_end": 1,
        "chunk_1_sha256": hashlib.sha256(b"evidence").hexdigest(),
        "chunk_2": "",
        "chunk_2_score": None,
        "chunk_2_source_index": -1,
        "chunk_2_window_index": -1,
        "chunk_2_token_start": -1,
        "chunk_2_token_end": -1,
        "chunk_2_sha256": "",
        "chunk_3": "",
        "chunk_3_score": None,
        "chunk_3_source_index": -1,
        "chunk_3_window_index": -1,
        "chunk_3_token_start": -1,
        "chunk_3_token_end": -1,
        "chunk_3_sha256": "",
        "chunk_4": "",
        "chunk_4_score": None,
        "chunk_4_source_index": -1,
        "chunk_4_window_index": -1,
        "chunk_4_token_start": -1,
        "chunk_4_token_end": -1,
        "chunk_4_sha256": "",
        "retriever_model": "mock",
        "retriever_revision": "v1",
        "tokenizer_model": "mock",
        "tokenizer_revision": "v1",
        "chunking_signature": "chunk-v1",
        "retrieval_signature": "retrieval-v1",
    }


def _parent(tmp_path: Path, rows: list[dict]) -> Path:
    run = tmp_path / "parent"
    run.mkdir()
    frame = pd.DataFrame(rows)
    frame.to_parquet(run / "dataset.parquet", index=False)
    retrieval = {"integrity": {"labels_used_for_claim_boundaries": False}}
    (run / "retrieval_audit.json").write_text(json.dumps(retrieval), encoding="utf-8")
    manifest = {
        "schema_version": PARENT_SCHEMA,
        "signature": "parent123",
        "counts": {"examples": len(frame)},
        "artifacts": {"dataset.parquet": sha256_file(run / "dataset.parquet")},
        "dataset": {"response_path": "missing-response.jsonl", "source_path": "missing-source.jsonl"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_group_id_preserves_exact_source_and_claim() -> None:
    assert duplicate_group_id("s1", "Claim") != duplicate_group_id("s2", "Claim")
    assert duplicate_group_id("s1", "Claim") != duplicate_group_id("s1", "claim")
    assert duplicate_group_id("s1", "Claim") == duplicate_group_id("s1", "Claim")


def test_input_equivalence_accepts_nan_and_close_scores() -> None:
    first = _row("s", "same claim", "r1", False)
    second = _row("s", "same claim", "r2", False)
    second["chunk_1_score"] = first["chunk_1_score"] + 1e-8
    assert input_differences(pd.DataFrame([first, second])) == []


def test_training_view_collapses_consistent_and_drops_conflict(tmp_path: Path) -> None:
    rows = [
        _row("s1", "same claim", "r2", False),
        _row("s1", "same claim", "r1", False),
        _row("s1", "conflicting claim", "r3", False),
        _row("s1", "conflicting claim", "r4", False),
        _row("s1", "conflicting claim", "r5", True),
        _row("s2", "same claim", "r6", False),
    ]
    parent = _parent(tmp_path, rows)
    result = build_training_view(parent, output_root=tmp_path / "out", policy=SUPPORTED_POLICY)
    output = tmp_path / "out" / SUPPORTED_POLICY / result["signature"]
    view = pd.read_parquet(output / "dataset.parquet")
    assert len(view) == 2
    assert set(view["source_id"]) == {"s1", "s2"}
    assert not view.duplicated(["source_id", "claim"]).any()
    assert len(pd.read_parquet(output / "excluded_conflicting_groups.parquet")) == 3
    assert result["schema_version"] == TRAINING_VIEW_SCHEMA
    assert result["parent"]["dataset_sha256"] == sha256_file(parent / "dataset.parquet")


def test_training_view_rejects_legacy_and_resume_validates_hash(tmp_path: Path) -> None:
    rows = [_row("s", "claim", "r", False)]
    parent = _parent(tmp_path, rows)
    result = build_training_view(parent, output_root=tmp_path / "out", policy=SUPPORTED_POLICY)
    output = tmp_path / "out" / SUPPORTED_POLICY / result["signature"]
    resumed = build_training_view(parent, output_root=tmp_path / "out", policy=SUPPORTED_POLICY, resume=True)
    assert resumed["signature"] == result["signature"]
    data = bytearray((output / "dataset.parquet").read_bytes())
    data[-1] ^= 1
    (output / "dataset.parquet").write_bytes(data)
    with pytest.raises(ValueError, match="manifesto inválido"):
        build_training_view(parent, output_root=tmp_path / "out", policy=SUPPORTED_POLICY, resume=True)
