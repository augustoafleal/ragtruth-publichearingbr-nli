
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .io_utils import iter_jsonl, sha256_file
from .ragtruth_data import (
    label_independent_sentence_claims,
    response_claims,
    sentence_spans,
)
from .ragtruth_top4_config import Top4Config
from .ragtruth_top4_embeddings import (
    Chunk,
    EmbeddingCache,
    OffsetTokenizer,
    WhitespaceTokenizer,
    build_encoder,
    build_tokenizer,
    chunk_passage,
    normalize_text,
    sha256_text,
    signature_for,
)


@dataclass
class RawTop4Example:
    example_id: str
    source_id: str
    response_id: str
    split: str
    claim: str
    label: bool
    claim_index: int
    claim_start: int
    claim_end: int
    question: str
    source_name: str
    generator_model: str
    temperature: Any
    label_types: list[str]
    span_texts: list[str]
    span_offsets: list[list[int]]
    candidates: list[Chunk]
    span_label_types: list[list[str]] = field(default_factory=list)
    span_overlap_chars: list[int] = field(default_factory=list)
    span_overlap_fraction_claim: list[float] = field(default_factory=list)
    span_overlap_fraction_spans: list[float] = field(default_factory=list)
    span_crosses_claim_boundary: bool = False
    claim_boundary_strategy: str = "label_dependent_sentences"
    response_text: str = ""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    if temporary.exists():
        temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _write_resume_parts(rows: list[dict[str, Any]], destination: Path, signature: str, batch_size: int = 256) -> None:
    parts_dir = destination / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for part_index, start in enumerate(range(0, len(rows), max(1, batch_size))):
        part_path = parts_dir / f"part-{part_index:06d}.jsonl"
        if not part_path.is_file():
            _atomic_jsonl(part_path, rows[start : start + max(1, batch_size)])
    _atomic_json(parts_dir / "manifest.json", {"schema_version": "ragtruth-top4-parts-v1", "signature": signature, "rows": len(rows), "parts": math.ceil(len(rows) / max(1, batch_size))})


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
    except ImportError as error:
        raise RuntimeError(
            "Parquet requer pyarrow. Instale com `pip install pyarrow` ou `pip install -e '.[dev,ragtruth]'`."
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_qa_passages_indexed(value: Any) -> list[tuple[int, str]]:
    if isinstance(value, dict):
        value = value.get("passages", "")
    if isinstance(value, list):
        result: list[tuple[int, str]] = []
        for index, item in enumerate(value):
            text = item.get("text", "") if isinstance(item, dict) else item
            text = normalize_text(str(text or ""))
            if text:
                result.append((index, text))
        return result
    text = str(value or "")
    if not text.strip():
        return []
    import re

    marker = re.compile(r"(?:^|\n\s*)passage\s+(\d+)\s*:\s*", re.IGNORECASE)
    matches = list(marker.finditer(text))
    if not matches:
        pieces = [normalize_text(part) for part in text.split("\n\n") if normalize_text(part)]
        return list(enumerate(pieces))
    passages: list[tuple[int, str]] = []
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        passage = normalize_text(text[start:end])
        if passage:
            passages.append((int(match.group(1)) - 1, passage))
    return passages


def _read_source_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(iter_jsonl(path))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if not source_id or source_id in result:
            raise ValueError(f"source_id ausente ou duplicado em {path}: {source_id!r}")
        result[source_id] = row
    return result


def _label_bounds(label: dict[str, Any], text_length: int) -> tuple[int, int] | None:
    try:
        start = max(0, min(int(label["start"]), text_length))
        end = max(0, min(int(label["end"]), text_length))
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _span_audit(response: dict[str, Any]) -> dict[str, int]:
    text = str(response.get("response", ""))
    bounds: list[tuple[int, int]] = []
    invalid = 0
    text_mismatch = 0
    for label in response.get("labels", []):
        if not isinstance(label, dict):
            invalid += 1
            continue
        span = _label_bounds(label, len(text))
        if span is None:
            invalid += 1
            continue
        bounds.append(span)
        if str(label.get("text", "")) and text[span[0] : span[1]] != str(label.get("text")):
            text_mismatch += 1
    ordered = sorted(bounds)
    overlap = sum(1 for left, right in zip(ordered, ordered[1:]) if right[0] < left[1])
    adjacent = sum(1 for left, right in zip(ordered, ordered[1:]) if right[0] == left[1])
    return {"spans": len(bounds), "invalid": invalid, "text_mismatch": text_mismatch, "overlap_pairs": overlap, "adjacent_pairs": adjacent}


def _claim_span_metadata(
    response_text: str,
    claim_start: int,
    claim_end: int,
    labels: list[dict[str, Any]],
    claim_intervals: list[tuple[int, int]],
) -> dict[str, Any]:
    span_texts: list[str] = []
    span_offsets: list[list[int]] = []
    span_label_types: list[list[str]] = []
    overlap_chars: list[int] = []
    overlap_fraction_claim: list[float] = []
    overlap_fraction_spans: list[float] = []
    crosses_boundary = False
    for label in labels:
        bounds = _label_bounds(label, len(response_text))
        if bounds is None:
            continue
        span_start, span_end = bounds
        intersection = max(0, min(claim_end, span_end) - max(claim_start, span_start))
        if intersection <= 0:
            continue
        touched_claims = sum(
            max(0, min(end, span_end) - max(start, span_start)) > 0
            for start, end in claim_intervals
        )
        crosses_boundary = crosses_boundary or touched_claims > 1
        span_texts.append(response_text[span_start:span_end])
        span_offsets.append([span_start, span_end])
        span_label_types.append([str(label.get("label_type", ""))])
        overlap_chars.append(int(intersection))
        overlap_fraction_claim.append(float(intersection / max(1, claim_end - claim_start)))
        overlap_fraction_spans.append(float(intersection / max(1, span_end - span_start)))
    return {
        "span_texts": span_texts,
        "span_offsets": span_offsets,
        "span_label_types": span_label_types,
        "span_overlap_chars": overlap_chars,
        "span_overlap_fraction_claim": overlap_fraction_claim,
        "span_overlap_fraction_spans": overlap_fraction_spans,
        "span_crosses_claim_boundary": bool(crosses_boundary),
    }


def _dataset_file_signature(response_path: Path, source_path: Path) -> dict[str, Any]:
    return {
        "response_path": str(response_path.resolve()),
        "source_path": str(source_path.resolve()),
        "response_sha256": sha256_file(response_path),
        "source_sha256": sha256_file(source_path),
    }


def _chunks_for_source(
    source_id: str,
    source_row: dict[str, Any],
    tokenizer: OffsetTokenizer,
    config: Top4Config,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source_index, passage in parse_qa_passages_indexed(source_row.get("source_info")):
        chunks.extend(
            chunk_passage(
                passage,
                source_index=source_index,
                tokenizer=tokenizer,
                chunk_size_tokens=config.retriever.chunk_size_tokens,
                chunk_overlap_tokens=config.retriever.chunk_overlap_tokens,
                source_id=source_id,
            )
        )
    return chunks


def load_qa_examples(
    config: Top4Config,
    tokenizer: OffsetTokenizer | None = None,
    limit: int | None = None,
    split_filter: set[str] | None = None,
) -> tuple[list[RawTop4Example], list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer or build_tokenizer(config.retriever)
    sources = _read_source_rows(config.dataset.source_path)
    source_chunks: dict[str, list[Chunk]] = {}
    rejected: list[dict[str, Any]] = []
    rows: list[RawTop4Example] = []
    seen_sources_split: dict[str, str] = {}
    response_count = 0
    span_totals = Counter()
    for response in iter_jsonl(config.dataset.response_path):
        response_count += 1
        response_id = str(response.get("id", ""))
        source_id = str(response.get("source_id", ""))
        source = sources.get(source_id)
        if source is None:
            rejected.append({"example_id": response_id, "source_id": source_id, "reason": "missing_source"})
            continue
        if str(source.get("task_type", "")) != config.dataset.task:
            continue
        split = str(response.get("split", ""))
        if split not in config.dataset.splits or (split_filter and split not in split_filter):
            continue
        previous_split = seen_sources_split.setdefault(source_id, split)
        if previous_split != split:
            raise ValueError(f"source_id {source_id} aparece em splits oficiais diferentes.")
        if str(response.get("quality", "good")) != config.dataset.quality:
            rejected.append({"example_id": response_id, "source_id": source_id, "split": split, "reason": f"quality:{response.get('quality')}"})
            continue
        span_totals.update(_span_audit(response))
        if config.dataset.boundary_strategy == "label_independent_sentences":
            if config.dataset.granularity != "claim":
                raise ValueError("label_independent_sentences exige granularity=claim")
            claims = label_independent_sentence_claims(response, min_words=config.dataset.min_words)
        else:
            claims = response_claims(response, granularity=config.dataset.granularity, min_words=config.dataset.min_words)
        if not claims:
            rejected.append({"example_id": response_id, "source_id": source_id, "split": split, "reason": "empty_claims"})
            continue
        if source_id not in source_chunks:
            source_chunks[source_id] = _chunks_for_source(source_id, source, tokenizer, config)
        candidates = source_chunks[source_id]
        if not candidates:
            rejected.append({"example_id": response_id, "source_id": source_id, "split": split, "reason": "no_candidate_chunks"})
            continue
        source_info = source.get("source_info") if isinstance(source.get("source_info"), dict) else {}
        response_text = str(response.get("response", ""))
        claim_intervals = [(int(item["claim_start"]), int(item["claim_end"])) for item in claims]
        seen_within_response: set[tuple[str, int, int]] = set()
        for claim_row in claims:
            occurrence_key = (str(claim_row["claim"]), int(claim_row["claim_start"]), int(claim_row["claim_end"]))
            if config.dataset.duplicate_policy == "deduplicate_within_response" and occurrence_key in seen_within_response:
                rejected.append({"example_id": response_id, "source_id": source_id, "split": split, "reason": "duplicate_within_response"})
                continue
            seen_within_response.add(occurrence_key)
            labels = [label for label in claim_row.get("labels", []) if isinstance(label, dict)]
            span_metadata = _claim_span_metadata(
                response_text,
                int(claim_row["claim_start"]),
                int(claim_row["claim_end"]),
                labels,
                claim_intervals,
            )
            if config.dataset.boundary_strategy == "label_independent_sentences":
                example_id = f"{response_id}:{claim_row['claim_start']}:{claim_row['claim_end']}"
            else:
                example_id = f"{response_id}:{claim_row['claim_index']}"
            rows.append(
                RawTop4Example(
                    example_id=example_id,
                    source_id=source_id,
                    response_id=response_id,
                    split=split,
                    claim=str(claim_row["claim"]),
                    label=bool(claim_row["label"]),
                    claim_index=int(claim_row["claim_index"]),
                    claim_start=int(claim_row["claim_start"]),
                    claim_end=int(claim_row["claim_end"]),
                    question=str(source_info.get("question", "")),
                    source_name=str(source.get("source", "")),
                    generator_model=str(response.get("model", "")),
                    temperature=response.get("temperature"),
                    label_types=sorted({str(label.get("label_type", "")) for label in labels}),
                    span_texts=span_metadata["span_texts"],
                    span_offsets=span_metadata["span_offsets"],
                    candidates=candidates,
                    span_label_types=span_metadata["span_label_types"],
                    span_overlap_chars=span_metadata["span_overlap_chars"],
                    span_overlap_fraction_claim=span_metadata["span_overlap_fraction_claim"],
                    span_overlap_fraction_spans=span_metadata["span_overlap_fraction_spans"],
                    span_crosses_claim_boundary=span_metadata["span_crosses_claim_boundary"],
                    claim_boundary_strategy=config.dataset.boundary_strategy,
                    response_text=response_text,
                )
            )
    rows.sort(key=lambda row: (row.split, row.source_id, row.response_id, row.claim_index, row.example_id))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    stats = {
        "raw_responses_seen": response_count,
        "parsed_examples": len(rows),
        "source_count": len({row.source_id for row in rows}),
        "candidate_chunk_count": len({chunk.chunk_key for row in rows for chunk in row.candidates}),
        "rejected_count": len(rejected),
        "span_audit": dict(span_totals),
    }
    return rows, rejected, stats


def _embedding_signature(config: Top4Config, dataset_signature: dict[str, Any], kind: str) -> str:
    payload: dict[str, Any] = {
        "kind": kind,
        "dataset": dataset_signature,
        "retriever": config.to_dict()["retriever"],
        "chunking": {
            "size": config.retriever.chunk_size_tokens,
            "overlap": config.retriever.chunk_overlap_tokens,
            "normalization": config.retriever.normalize_whitespace,
        },
    }
    if kind == "claims":
        payload["schema_version"] = config.schema_version
        payload["boundary_strategy"] = config.dataset.boundary_strategy
        payload["segmentation_version"] = config.dataset.segmentation_version
        payload["overlap_rule"] = config.dataset.overlap_rule
    return signature_for(payload)


def _legacy_v1_embedding_signature(config: Top4Config, dataset_signature: dict[str, Any], kind: str) -> str:
    return signature_for({
        "kind": kind,
        "schema": "ragtruth-qa-top4-v1",
        "dataset": dataset_signature,
        "retriever": config.to_dict()["retriever"],
        "chunking": {"size": config.retriever.chunk_size_tokens, "overlap": config.retriever.chunk_overlap_tokens},
    })


def _encode_cached(
    keys: list[str],
    texts: list[str],
    prefix: str,
    cache: EmbeddingCache,
    encoder: Any,
    batch_size: int,
    fallback_caches: list[EmbeddingCache] | None = None,
) -> tuple[np.ndarray, bool, str]:
    cached = cache.load(keys, encoder.dimension)
    if cached is not None:
        return cached, True, str(cache.data_path)
    for fallback in fallback_caches or []:
        cached = fallback.load(keys, encoder.dimension)
        if cached is not None:
            cache.save(keys, cached)
            return cached, True, str(fallback.data_path)
    embeddings = encoder.encode(texts, prefix=prefix, batch_size=batch_size)
    if embeddings.shape != (len(keys), encoder.dimension):
        raise ValueError(f"Dimensão de embedding inesperada: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contêm NaN ou infinito.")
    norms = np.linalg.norm(embeddings, axis=1)
    if len(norms) and not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("Embeddings não estão normalizados em L2.")
    cache.save(keys, embeddings)
    return embeddings, False, str(cache.data_path)


def _score_examples(
    examples: list[RawTop4Example],
    chunk_embeddings: dict[str, np.ndarray],
    claim_embeddings: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        claim_key = sha256_text(example.claim)
        claim_vector = claim_embeddings[claim_key]
        scored = []
        for chunk in example.candidates:
            score = float(np.dot(claim_vector, chunk_embeddings[chunk.chunk_key]))
            scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].source_index, pair[1].window_index, pair[1].chunk_key))
        selected = scored[:4]
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "source_id": example.source_id,
            "response_id": example.response_id,
            "split": example.split,
            "task": "QA",
            "claim": example.claim,
            "label": example.label,
            "claim_index": example.claim_index,
            "claim_start": example.claim_start,
            "claim_end": example.claim_end,
            "question": example.question,
            "source_name": example.source_name,
            "generator_model": example.generator_model,
            "model": example.generator_model,
            "temperature": example.temperature,
            "label_types": example.label_types,
            "span_texts": example.span_texts or [],
            "span_offsets": example.span_offsets or [],
            "span_label_types": example.span_label_types or [],
            "span_overlap_chars": example.span_overlap_chars or [],
            "span_overlap_fraction_claim": example.span_overlap_fraction_claim or [],
            "span_overlap_fraction_spans": example.span_overlap_fraction_spans or [],
            "span_crosses_claim_boundary": bool(example.span_crosses_claim_boundary),
            "claim_boundary_strategy": example.claim_boundary_strategy,
            "label_span_count": len(example.span_offsets),
            "num_candidate_chunks": len(example.candidates),
            "evidence_mask": [],
        }
        mask: list[bool] = []
        for slot in range(4):
            if slot < len(selected):
                score, chunk = selected[slot]
                row[f"chunk_{slot + 1}"] = chunk.text
                row[f"chunk_{slot + 1}_score"] = score
                row[f"chunk_{slot + 1}_source_index"] = chunk.source_index
                row[f"chunk_{slot + 1}_window_index"] = chunk.window_index
                row[f"chunk_{slot + 1}_token_start"] = chunk.token_start
                row[f"chunk_{slot + 1}_token_end"] = chunk.token_end
                row[f"chunk_{slot + 1}_sha256"] = chunk.text_sha256
                mask.append(True)
            else:
                row[f"chunk_{slot + 1}"] = ""
                row[f"chunk_{slot + 1}_score"] = None
                row[f"chunk_{slot + 1}_source_index"] = -1
                row[f"chunk_{slot + 1}_window_index"] = -1
                row[f"chunk_{slot + 1}_token_start"] = -1
                row[f"chunk_{slot + 1}_token_end"] = -1
                row[f"chunk_{slot + 1}_sha256"] = ""
                mask.append(False)
        row["evidence_mask"] = mask
        rows.append(row)
    return rows


def _add_run_signatures(rows: list[dict[str, Any]], chunking_signature: str, retrieval_signature: str, retriever: Any, tokenizer: Any) -> None:
    for row in rows:
        row["retriever_model"] = retriever.model_id
        row["retriever_revision"] = retriever.revision
        row["tokenizer_model"] = tokenizer.model_id
        row["tokenizer_revision"] = tokenizer.revision
        row["chunking_signature"] = chunking_signature
        row["retrieval_signature"] = retrieval_signature


def _near_duplicate(a: str, b: str) -> bool:
    left, right = set(normalize_text(a).lower().split()), set(normalize_text(b).lower().split())
    if not left or not right:
        return False
    return len(left & right) / max(1, len(left | right)) >= 0.9


def build_audit(rows: list[dict[str, Any]], rejected: list[dict[str, Any]], config: Top4Config) -> dict[str, Any]:
    split_label = Counter((str(row["split"]), int(bool(row["label"]))) for row in rows)
    split_sources: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_sources[str(row["split"])].add(str(row["source_id"]))
    overlap = sum(1 for left in split_sources for right in split_sources if left < right and split_sources[left] & split_sources[right])
    candidate_counts = [int(row["num_candidate_chunks"]) for row in rows]
    scores = [[row[f"chunk_{i}_score"] for i in range(1, 5) if row[f"chunk_{i}_score"] is not None] for row in rows]
    top1 = [values[0] for values in scores if values]
    top4 = [values[-1] for values in scores if values]
    identical = 0
    near_duplicate = 0
    source_overlap = 0
    for row in rows:
        texts = [row[f"chunk_{i}"] for i in range(1, 5) if row[f"evidence_mask"][i - 1]]
        if len(texts) != len(set(texts)):
            identical += 1
        if any(_near_duplicate(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))):
            near_duplicate += 1
        indices = [row[f"chunk_{i}_source_index"] for i in range(1, 5) if row[f"evidence_mask"][i - 1]]
        if len(indices) != len(set(indices)):
            source_overlap += 1
    integrity_errors: list[str] = []
    ids = [str(row["example_id"]) for row in rows]
    claim_source_pairs = [(str(row["source_id"]), str(row["claim"])) for row in rows]
    claim_labels: dict[tuple[str, str], set[bool]] = defaultdict(set)
    for row in rows:
        claim_labels[(str(row["source_id"]), str(row["claim"]))].add(bool(row["label"]))
    if len(ids) != len(set(ids)):
        integrity_errors.append("duplicate_example_id")
    for row in rows:
        if len(row["evidence_mask"]) != 4:
            integrity_errors.append(f"mask_length:{row['example_id']}")
        values = [row[f"chunk_{i}_score"] for i in range(1, 5) if row[f"chunk_{i}_score"] is not None]
        if any(values[i] < values[i + 1] - 1e-7 for i in range(len(values) - 1)):
            integrity_errors.append(f"score_order:{row['example_id']}")
        for i in range(1, 5):
            valid = bool(row["evidence_mask"][i - 1])
            if valid != bool(row[f"chunk_{i}"]):
                integrity_errors.append(f"mask_text:{row['example_id']}:{i}")
            if not valid and row[f"chunk_{i}_source_index"] != -1:
                integrity_errors.append(f"mask_index:{row['example_id']}:{i}")
    stats = {
        "dataset": {
            "examples": len(rows),
            "examples_by_split": dict(Counter(str(row["split"]) for row in rows)),
            "examples_by_label": dict(Counter(int(bool(row["label"])) for row in rows)),
            "prevalence_by_split": {
                split: float(np.mean([bool(row["label"]) for row in rows if row["split"] == split]))
                for split in sorted({str(row["split"]) for row in rows})
            },
            "unique_sources": len({row["source_id"] for row in rows}),
            "rejected_examples": len(rejected),
            "claim_words": {
                "min": min((len(str(row["claim"]).split()) for row in rows), default=0),
                "max": max((len(str(row["claim"]).split()) for row in rows), default=0),
                "mean": float(np.mean([len(str(row["claim"]).split()) for row in rows])) if rows else 0.0,
            },
            "passage_chunks": {
                "min": min(candidate_counts, default=0),
                "max": max(candidate_counts, default=0),
                "mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
                "counts": dict(Counter(str(x) if x < 4 else "4+" for x in candidate_counts)),
            },
        },
        "retrieval": {
            "top1_score": {"min": min(top1, default=0.0), "max": max(top1, default=0.0), "mean": float(np.mean(top1)) if top1 else 0.0},
            "top4_score": {"min": min(top4, default=0.0), "max": max(top4, default=0.0), "mean": float(np.mean(top4)) if top4 else 0.0},
            "top1_minus_top4": {"mean": float(np.mean([a - b for a, b in zip(top1, top4)])) if top1 else 0.0},
            "masked_slot_rate": float(sum(not mask for row in rows for mask in row["evidence_mask"]) / max(1, len(rows) * 4)),
            "identical_top4_rate": identical / max(1, len(rows)),
            "near_duplicate_top4_rate": near_duplicate / max(1, len(rows)),
            "same_source_index_top4_rate": source_overlap / max(1, len(rows)),
            "by_split_label": {f"{split}:{label}": count for (split, label), count in split_label.items()},
        },
        "integrity": {
            "source_isolation_verified_by_construction": True,
            "labels_used_for_claim_boundaries": config.dataset.boundary_strategy != "label_independent_sentences",
            "labels_used_for_retrieval": False,
            "duplicate_example_ids": len(ids) != len(set(ids)),
            "duplicate_claims_within_source": len(claim_source_pairs) - len(set(claim_source_pairs)),
            "conflicting_duplicate_claims": sum(1 for labels in claim_labels.values() if len(labels) > 1),
            "label_semantics": "1 means a claim overlaps at least one RAGTruth hallucination annotation span; 0 means no annotation span overlaps the claim.",
            "split_source_overlap_count": overlap,
            "exactly_four_slots": all(len(row["evidence_mask"]) == 4 for row in rows),
            "errors": sorted(set(integrity_errors)),
        },
        "evidence_coverage": {
            "available": False,
            "reason": "RAGTruth QA não fornece alinhamento de evidência anotada no source_info para Recall@k.",
        },
    }
    if overlap:
        stats["integrity"]["errors"].append("split_source_overlap")
    if stats["integrity"]["errors"]:
        raise ValueError(f"Falhas de integridade no dataset top-4: {stats['integrity']['errors']}")
    return stats


def _surface_row(row: dict[str, Any]) -> dict[str, Any]:
    import re

    text = str(row.get("claim", ""))
    words = re.findall(r"\b\w+(?:['’-]\w+)*\b", text, flags=re.UNICODE)
    sentences = len(sentence_spans(text)) if text.strip() else 0
    return {
        "words": len(words),
        "chars": len(text),
        "sentences": sentences,
        "multi_sentence": sentences > 1,
        "enumeration": bool(re.search(r"(?:^|\n|\s)(?:\*|[-•]|\(?\d+[.)]|[A-Za-z][.)])\s", text)),
        "numbers": bool(re.search(r"\d", text)),
        "quotes": bool(re.search(r'[\"“”‘’]', text)),
        "negation": bool(re.search(r"\b(?:not|no|never|without|cannot|can't|didn't|isn't|aren't|won't|não|nunca|sem|nem)\b", text, re.I)),
        "terminal_punctuation": bool(re.search(r"[.!?؟。！？]\s*$", text.strip())),
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _cohen_d(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.0
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    pos_var = float(np.var(pos, ddof=1)) if len(pos) > 1 else 0.0
    neg_var = float(np.var(neg, ddof=1)) if len(neg) > 1 else 0.0
    denominator = np.sqrt(((len(pos) - 1) * pos_var + (len(neg) - 1) * neg_var) / max(1, len(pos) + len(neg) - 2))
    return float((np.mean(pos) - np.mean(neg)) / denominator) if denominator else 0.0


def build_boundary_audit(rows: list[dict[str, Any]], rejected: list[dict[str, Any]], config: Top4Config) -> dict[str, Any]:
    enriched = [{**row, **_surface_row(row)} for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[f"{row['split']}:{int(bool(row['label']))}"].append(row)
    distributions: dict[str, Any] = {}
    numeric = ["chars", "words", "sentences", "multi_sentence", "span_overlap_fraction_claim", "label_span_count"]
    for key, values in sorted(groups.items()):
        data: dict[str, Any] = {}
        for field in numeric:
            if field == "span_overlap_fraction_claim":
                numbers = [max(row.get(field) or [0.0]) if row.get(field) else 0.0 for row in values]
            else:
                numbers = [float(row.get(field, 0)) for row in values]
            data[field] = _distribution(numbers)
        for field in ["enumeration", "numbers", "quotes", "negation", "terminal_punctuation"]:
            data[field + "_rate"] = float(np.mean([bool(row[field]) for row in values])) if values else 0.0
        distributions[key] = data
    positive = [row for row in enriched if bool(row["label"])]
    negative = [row for row in enriched if not bool(row["label"])]
    duplicate_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        duplicate_groups[(str(row["source_id"]), str(row["claim"]))].append(row)
    span_crossings = sum(bool(row.get("span_crosses_claim_boundary")) for row in rows)
    intervals_by_response: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        intervals_by_response[str(row["response_id"])].append((int(row["claim_start"]), int(row["claim_end"])))
    span_crossing_spans = 0
    seen_spans: set[tuple[str, int, int]] = set()
    for row in rows:
        for offsets in (row.get("span_offsets") or []):
            span_key = (str(row["response_id"]), int(offsets[0]), int(offsets[1]))
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            touched = sum(
                max(0, min(end, offsets[1]) - max(start, offsets[0])) > 0
                for start, end in intervals_by_response[str(row["response_id"])]
            )
            if touched > 1:
                span_crossing_spans += 1
    return {
        "boundary_strategy": config.dataset.boundary_strategy,
        "segmentation_version": config.dataset.segmentation_version,
        "overlap_rule": config.dataset.overlap_rule,
        "labels_used_for_claim_boundaries": config.dataset.boundary_strategy != "label_independent_sentences",
        "claims": len(rows),
        "responses": len({row["response_id"] for row in rows}),
        "claims_per_response": _distribution([count for count in Counter(row["response_id"] for row in rows).values()]),
        "claims_by_label": dict(Counter(int(bool(row["label"])) for row in rows)),
        "prevalence_by_split": {
            split: float(np.mean([bool(row["label"]) for row in rows if row["split"] == split]))
            for split in sorted({str(row["split"]) for row in rows})
        },
        "claims_touched_by_multiple_spans": sum(len(row.get("span_offsets") or []) > 1 for row in rows),
        "spans_crossing_claim_boundary": span_crossing_spans,
        "claims_marked_positive_by_crossing_span": span_crossings,
        "claims_empty_or_rejected": len(rejected),
        "duplicate_groups": sum(len(values) > 1 for values in duplicate_groups.values()),
        "conflicting_duplicate_groups": sum(len({bool(row["label"]) for row in values}) > 1 for values in duplicate_groups.values()),
        "distributions": distributions,
        "standardized_effects": {
            "words": _cohen_d([row["words"] for row in positive], [row["words"] for row in negative]),
            "chars": _cohen_d([row["chars"] for row in positive], [row["chars"] for row in negative]),
            "sentences": _cohen_d([row["sentences"] for row in positive], [row["sentences"] for row in negative]),
            "multi_sentence": _cohen_d([row["multi_sentence"] for row in positive], [row["multi_sentence"] for row in negative]),
        },
        "invariant": "For the same response, changing labels may change label/span metadata only; claim text, offsets, count and IDs remain unchanged.",
    }


def build_duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source_id"]), str(row["claim"]))].append(row)
    repeated = {key: values for key, values in groups.items() if len(values) > 1}
    details = []
    for (source_id, claim), values in sorted(repeated.items()):
        details.append({
            "source_id": source_id,
            "claim": claim,
            "multiplicity": len(values),
            "same_response": len({row["response_id"] for row in values}) == 1,
            "models": sorted({row.get("generator_model", "") for row in values}),
            "labels": sorted({int(bool(row["label"])) for row in values}),
            "example_ids": [row["example_id"] for row in values],
            "conflict": len({bool(row["label"]) for row in values}) > 1,
        })
    return {
        "policy": "keep_all",
        "groups_total": len(groups),
        "groups_repeated": len(repeated),
        "duplicate_rows": int(sum(len(values) for values in repeated.values())),
        "extra_rows": int(sum(len(values) - 1 for values in repeated.values())),
        "max_multiplicity": max((len(values) for values in groups.values()), default=0),
        "same_response_groups": sum(len({row["response_id"] for row in values}) == 1 for values in repeated.values()),
        "cross_response_groups": sum(len({row["response_id"] for row in values}) > 1 for values in repeated.values()),
        "cross_model_groups": sum(len({row.get("generator_model", "") for row in values}) > 1 for values in repeated.values()),
        "conflicting_groups": sum(len({bool(row["label"]) for row in values}) > 1 for values in repeated.values()),
        "groups": details,
    }


def build_surface_feature_audit(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"status": "skipped", "reason": "scikit-learn não disponível"}
    if len({bool(row["label"]) for row in rows}) < 2:
        return {"status": "skipped", "reason": "amostra sem as duas classes"}
    feature_names = ["words", "chars", "sentences", "enumeration", "numbers", "quotes", "negation", "terminal_punctuation"]
    enriched = [{**row, **_surface_row(row)} for row in rows if row["split"] == "train"]
    source_labels: dict[str, int] = {}
    for row in enriched:
        source_labels[str(row["source_id"])] = max(source_labels.get(str(row["source_id"]), 0), int(bool(row["label"])))
    source_ids = sorted(source_labels)
    if len(source_ids) < 4 or len(set(source_labels.values())) < 2:
        return {"status": "skipped", "reason": "fontes insuficientes para validação agrupada"}
    train_sources, validation_sources = train_test_split(
        source_ids,
        test_size=0.15,
        random_state=seed,
        stratify=[source_labels[source_id] for source_id in source_ids],
    )
    train_set, validation_set = set(train_sources), set(validation_sources)
    train_rows = [row for row in enriched if str(row["source_id"]) in train_set]
    validation_rows = [row for row in enriched if str(row["source_id"]) in validation_set]
    x_train = np.asarray([[float(row[name]) for name in feature_names] for row in train_rows])
    y_train = np.asarray([int(bool(row["label"])) for row in train_rows])
    x_validation = np.asarray([[float(row[name]) for name in feature_names] for row in validation_rows])
    y_validation = np.asarray([int(bool(row["label"])) for row in validation_rows])
    if len(set(y_train)) < 2 or len(set(y_validation)) < 2:
        return {"status": "skipped", "reason": "uma divisão agrupada ficou sem as duas classes"}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_validation)[:, 1]
    return {
        "status": "completed",
        "features": feature_names,
        "group_key": "source_id",
        "seed": seed,
        "train_sources": len(train_set),
        "validation_sources": len(validation_set),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_prevalence": float(np.mean(y_train)),
        "validation_prevalence": float(np.mean(y_validation)),
        "validation_AUPRC": float(average_precision_score(y_validation, probabilities)),
        "validation_AUROC": float(roc_auc_score(y_validation, probabilities)),
        "prevalence_baseline_AUPRC": float(np.mean(y_validation)),
        "purpose": "auditoria de leakage superficial; não é resultado do classificador do projeto",
    }


def build_legacy_comparison(rows: list[dict[str, Any]], legacy_dir: Path | None) -> dict[str, Any]:
    if legacy_dir is None or not (legacy_dir / "dataset.parquet").is_file():
        return {"status": "unavailable", "reason": "run anterior não encontrado"}
    legacy = pd.read_parquet(legacy_dir / "dataset.parquet")
    current = pd.DataFrame(rows)
    old_keys = set(zip(legacy.response_id.astype(str), legacy.claim_start.astype(int), legacy.claim_end.astype(int)))
    new_keys = set(zip(current.response_id.astype(str), current.claim_start.astype(int), current.claim_end.astype(int)))
    old_claim_text = set(legacy.claim.astype(str))
    new_claim_text = set(current.claim.astype(str))
    old_counts = legacy.groupby("response_id").size()
    new_counts = current.groupby("response_id").size()
    split_claims = sum(int(new_counts.get(response_id, 0) > old_counts.get(response_id, 0)) for response_id in new_counts.index)
    return {
        "status": "completed",
        "legacy_dir": str(legacy_dir),
        "legacy_examples": int(len(legacy)),
        "new_examples": int(len(current)),
        "legacy_positive": int(legacy.label.astype(bool).sum()),
        "new_positive": int(current.label.astype(bool).sum()),
        "exact_boundary_matches": int(len(old_keys & new_keys)),
        "exact_claim_text_matches": int(len(old_claim_text & new_claim_text)),
        "responses_with_more_claims": int(split_claims),
        "legacy_duplicate_groups": int(legacy.groupby(["source_id", "claim"]).size().gt(1).sum()),
        "new_duplicate_groups": int(current.groupby(["source_id", "claim"]).size().gt(1).sum()),
        "old_signature": "507bfd0a5ed06f92",
    }


def _write_audit_sample(rows: list[dict[str, Any]], output_path: Path, sample_size: int, seed: int = 42) -> None:
    if not rows:
        _atomic_text(output_path, "")
        return
    by_group: dict[tuple[str, int, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        top1 = float(row["chunk_1_score"]) if row["chunk_1_score"] is not None else -1.0
        band = "low" if top1 < 0.25 else ("mid" if top1 < 0.6 else "high")
        words = len(str(row["claim"]).split())
        length_band = "short" if words <= 8 else ("medium" if words <= 30 else "long")
        candidates = int(row["num_candidate_chunks"])
        candidate_band = "lt4" if candidates < 4 else "ge4"
        scores = [row[f"chunk_{i}_score"] for i in range(1, 5) if row[f"chunk_{i}_score"] is not None]
        gap_band = "large_gap" if len(scores) >= 2 and scores[0] - scores[-1] >= 0.25 else "regular_gap"
        span_band = "cross_boundary" if bool(row.get("span_crosses_claim_boundary")) else ("multi_span" if int(row.get("label_span_count", 0)) > 1 else ("single_span" if row.get("label_span_count", 0) else "no_span"))
        sentence_band = "multi_sentence" if len(sentence_spans(str(row.get("claim", "")))) > 1 else "single_sentence"
        by_group[(str(row["split"]), int(bool(row["label"])), band, length_band, candidate_band + ":" + gap_band, span_band, sentence_band)].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    groups = sorted(by_group)
    while len(selected) < min(sample_size, len(rows)) and groups:
        progressed = False
        for group in groups:
            if by_group[group]:
                selected.append(by_group[group].pop(rng.randrange(len(by_group[group]))))
                progressed = True
                if len(selected) >= sample_size:
                    break
        if not progressed:
            break
    selected.sort(key=lambda row: str(row["example_id"]))
    fields = [
        "example_id", "source_id", "response_id", "split", "task", "claim", "label", "claim_index", "claim_start", "claim_end",
        "label_span_count", "span_texts", "span_offsets", "span_label_types", "span_overlap_chars", "span_overlap_fraction_claim",
        "span_overlap_fraction_spans", "span_crosses_claim_boundary", "claim_boundary_strategy", "evidence_mask",
        "top4_sufficient", "support_present", "contradiction_present", "ambiguous", "annotation_confidence",
        "boundary_correct", "label_correct_given_span", "notes",
        "chunk_1", "chunk_1_score", "chunk_1_source_index", "chunk_2", "chunk_2_score", "chunk_2_source_index",
        "chunk_3", "chunk_3_score", "chunk_3_source_index", "chunk_4", "chunk_4_score", "chunk_4_source_index",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", mode="w", encoding="utf-8", newline="", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            values = {field: row.get(field, "") for field in fields}
            for field in [
                "evidence_mask", "span_texts", "span_offsets", "span_label_types", "span_overlap_chars",
                "span_overlap_fraction_claim", "span_overlap_fraction_spans",
            ]:
                values[field] = json.dumps(row.get(field, []), ensure_ascii=False)
            writer.writerow(values)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, output_path)


def _environment(encoder: Any) -> dict[str, Any]:
    import platform
    import sys

    versions: dict[str, str | None] = {}
    for package in ["torch", "transformers", "sentence-transformers", "numpy", "pandas", "datasets"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    if torch_cuda_available := bool(getattr(__import__("torch"), "cuda").is_available()):
        torch_module = __import__("torch")
        gpu = {"name": torch_module.cuda.get_device_name(0), "vram_bytes": torch_module.cuda.get_device_properties(0).total_memory}
    return {"python": sys.version, "platform": platform.platform(), "versions": versions, "device": getattr(encoder, "device", "cpu"), "gpu": gpu, "cuda_available": torch_cuda_available}


def _resolved_output_dir(config: Top4Config, output_dir: Path | None, signature: str) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (config.output_root / config.run_name / signature).resolve()


def _valid_existing_manifest(path: Path, signature: str, schema_version: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature or manifest.get("schema_version") != schema_version:
            return None
        for name, expected_hash in dict(manifest.get("artifacts", {})).items():
            artifact = path.parent / str(name)
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                return None
        if not (path.parent / "dataset.parquet").is_file():
            return None
        return manifest
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prepare_top4(
    config: Top4Config,
    *,
    output_dir: Path | None = None,
    split_filter: set[str] | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if device or batch_size:
        config = replace(
            config,
            retriever=replace(
                config.retriever,
                device=device or config.retriever.device,
                batch_size=int(batch_size or config.retriever.batch_size),
            ),
        )
    dataset_signature = _dataset_file_signature(config.dataset.response_path, config.dataset.source_path)
    selected_split = sorted(split_filter or set(config.dataset.splits))
    run_signature = signature_for({"config": config.to_dict(), "dataset": dataset_signature, "split": selected_split, "limit": limit})
    destination = _resolved_output_dir(config, output_dir, run_signature)
    destination.mkdir(parents=True, exist_ok=True)
    final_manifest = destination / "manifest.json"
    if resume and not force:
        existing = _valid_existing_manifest(final_manifest, run_signature, config.schema_version)
        if existing is not None:
            return existing
    tokenizer = build_tokenizer(config.retriever)
    examples, rejected, parse_stats = load_qa_examples(config, tokenizer=tokenizer, limit=limit, split_filter=set(selected_split))
    parse_seconds = time.perf_counter() - started
    if not examples:
        raise ValueError("Nenhum exemplo QA elegível foi encontrado.")
    chunk_by_key = {chunk.chunk_key: chunk for example in examples for chunk in example.candidates}
    chunk_keys = sorted(chunk_by_key)
    claim_keys = sorted({sha256_text(example.claim) for example in examples})
    claim_texts = {sha256_text(example.claim): example.claim for example in examples}
    retriever_dict = config.to_dict()["retriever"]
    chunking_signature = signature_for({"schema": config.schema_version, "tokenizer": retriever_dict["tokenizer_model_id"], "tokenizer_revision": retriever_dict["tokenizer_revision"], "size": config.retriever.chunk_size_tokens, "overlap": config.retriever.chunk_overlap_tokens, "normalization": config.retriever.normalize_whitespace})
    retrieval_signature = signature_for({"schema": config.schema_version, "dataset": dataset_signature, "retriever": retriever_dict, "strategy": config.retriever.strategy, "chunking_signature": chunking_signature})
    encoder = build_encoder(config.retriever)
    if str(getattr(encoder, "device", "cpu")).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    chunk_cache = EmbeddingCache(config.cache_root, "chunks", _embedding_signature(config, dataset_signature, "chunks"))
    claim_cache = EmbeddingCache(config.cache_root, "claims", _embedding_signature(config, dataset_signature, "claims"))
    legacy_chunk_cache = EmbeddingCache(config.cache_root, "chunks", _legacy_v1_embedding_signature(config, dataset_signature, "chunks"))
    cache_reuse: dict[str, Any] = {"chunks": {"reused": False, "path": str(chunk_cache.data_path)}, "claims": {"reused": False, "path": str(claim_cache.data_path)}}
    fallback_caches = [legacy_chunk_cache] if legacy_chunk_cache.signature != chunk_cache.signature else []
    chunk_embeddings_array, chunk_reused, chunk_cache_source = _encode_cached(chunk_keys, [chunk_by_key[key].text for key in chunk_keys], config.retriever.prefix_passage, chunk_cache, encoder, config.retriever.batch_size, fallback_caches=fallback_caches)
    cache_reuse["chunks"]["reused"] = chunk_reused
    cache_reuse["chunks"]["path"] = chunk_cache_source
    cache_reuse["chunks"]["copied_to"] = str(chunk_cache.data_path) if chunk_cache_source != str(chunk_cache.data_path) else None
    claim_embeddings_array, claim_reused, claim_cache_source = _encode_cached(claim_keys, [claim_texts[key] for key in claim_keys], config.retriever.prefix_query, claim_cache, encoder, config.retriever.batch_size)
    cache_reuse["claims"]["reused"] = claim_reused
    cache_reuse["claims"]["path"] = claim_cache_source
    embedding_seconds = time.perf_counter() - started - parse_seconds
    chunk_embeddings = dict(zip(chunk_keys, chunk_embeddings_array, strict=True))
    claim_embeddings = dict(zip(claim_keys, claim_embeddings_array, strict=True))
    rows = _score_examples(examples, chunk_embeddings, claim_embeddings)
    _add_run_signatures(rows, chunking_signature, retrieval_signature, encoder, tokenizer)
    audit = build_audit(rows, rejected, config)
    boundary_audit = build_boundary_audit(rows, rejected, config)
    duplicate_audit = build_duplicate_audit(rows)
    surface_audit = build_surface_feature_audit(rows, seed=config.audit_seed)
    legacy_comparison = build_legacy_comparison(rows, config.legacy_run_dir)
    total_seconds = time.perf_counter() - started
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if str(getattr(encoder, "device", "cpu")).startswith("cuda") else 0
    _write_resume_parts(rows, destination, run_signature)
    _atomic_parquet(destination / "dataset.parquet", pd.DataFrame(rows))
    _atomic_jsonl(destination / "rejected_examples.jsonl", rejected)
    _write_audit_sample(rows, destination / "audit_sample.csv", config.sample_size, seed=config.audit_seed)
    _atomic_json(destination / "counts.json", {**parse_stats, **audit["dataset"]})
    _atomic_json(destination / "retrieval_audit.json", audit)
    _atomic_json(destination / "boundary_audit.json", boundary_audit)
    _atomic_json(destination / "duplicate_audit.json", duplicate_audit)
    _atomic_json(destination / "surface_feature_audit.json", surface_audit)
    _atomic_json(destination / "legacy_comparison.json", legacy_comparison)
    _atomic_json(destination / "resolved_config.json", config.to_dict())
    _atomic_jsonl(destination / "run_log.jsonl", [{"event": "prepare_completed", "signature": run_signature, "rows": len(rows), "chunks": len(chunk_keys), "claims": len(claim_keys), "parse_seconds": parse_seconds, "embedding_seconds": embedding_seconds, "total_seconds": total_seconds, "peak_gpu_bytes": peak_gpu_bytes}])
    artifacts = {name: sha256_file(destination / name) for name in ["dataset.parquet", "rejected_examples.jsonl", "audit_sample.csv", "counts.json", "retrieval_audit.json", "boundary_audit.json", "duplicate_audit.json", "surface_feature_audit.json", "legacy_comparison.json", "resolved_config.json", "run_log.jsonl"]}
    manifest = {
        "schema_version": config.schema_version,
        "signature": run_signature,
        "dataset": {"name": "RAGTruth", "revision": config.dataset.revision, "task": "QA", **dataset_signature, "splits": selected_split, "quality": config.dataset.quality},
        "counts": {"examples": len(rows), "claims": len(rows), "unique_sources": len({row["source_id"] for row in rows}), "unique_chunks": len(chunk_keys), "unique_claim_embeddings": len(claim_keys), "rejected": len(rejected), "by_split": dict(Counter(row["split"] for row in rows)), "by_label": dict(Counter(int(bool(row["label"])) for row in rows)), "span_audit": parse_stats.get("span_audit", {})},
        "retriever": {"model": encoder.model_id, "revision": encoder.revision, "tokenizer": tokenizer.model_id, "tokenizer_revision": tokenizer.revision, "embedding_dimension": encoder.dimension, "dtype": encoder.dtype, "prefix_query": config.retriever.prefix_query, "prefix_passage": config.retriever.prefix_passage, "pooling": "attention_mask_mean_then_l2_normalize", "strategy": config.retriever.strategy, "chunk_size_tokens": config.retriever.chunk_size_tokens, "chunk_overlap_tokens": config.retriever.chunk_overlap_tokens, "chunking_signature": chunking_signature, "retrieval_signature": retrieval_signature},
        "environment": _environment(encoder),
        "timings_seconds": {"parse_and_chunk": parse_seconds, "embedding": embedding_seconds, "total": total_seconds},
        "peak_gpu_bytes": peak_gpu_bytes,
        "audit": audit,
        "boundary_audit": boundary_audit,
        "duplicate_audit": {key: value for key, value in duplicate_audit.items() if key != "groups"},
        "surface_feature_audit": surface_audit,
        "legacy_comparison": legacy_comparison,
        "artifacts": artifacts,
        "cache": {"root": str(config.cache_root.resolve()), "chunk_cache": str(chunk_cache.data_path), "claim_cache": str(claim_cache.data_path), "signature_checked": True, "reuse": cache_reuse, "claim_cache_reused_from_legacy": False},
    }
    _atomic_json(final_manifest, manifest)
    return manifest


def estimate_top4(config: Top4Config, *, split_filter: set[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    tokenizer = WhitespaceTokenizer()
    examples, rejected, stats = load_qa_examples(config, tokenizer=tokenizer, limit=limit, split_filter=split_filter)
    chunk_count = len({chunk.chunk_key for row in examples for chunk in row.candidates})
    claim_count = len({sha256_text(row.claim) for row in examples})
    dim = config.retriever.embedding_dim
    return {"encoder_loaded": False, "embeddings_computed": False, **stats, "examples": len(examples), "unique_chunks": chunk_count, "unique_claims": claim_count, "estimated_embedding_bytes": (chunk_count + claim_count) * dim * 4, "estimated_embedding_mib": (chunk_count + claim_count) * dim * 4 / 2**20, "estimated_batches": math.ceil(chunk_count / config.retriever.batch_size) + math.ceil(claim_count / config.retriever.batch_size), "rejected_by_reason": dict(Counter(row.get("reason", "unknown") for row in rejected))}
