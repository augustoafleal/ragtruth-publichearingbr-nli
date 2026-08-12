from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from .io_utils import read_jsonl, sha256_file, write_json, write_jsonl

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’\]\)]*)\s+(?=[A-Z0-9\"'“‘\[(])")
_PASSAGE_MARKER = re.compile(r"(?:^|\n\s*)passage\s+\d+\s*:\s*", re.IGNORECASE)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    text = str(text)
    if not text.strip():
        return []
    starts = [0]
    for match in _SENTENCE_BOUNDARY.finditer(text):
        starts.append(match.end())
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans or [(0, len(text))]


def _label_span(label: dict[str, Any], text_length: int) -> tuple[int, int] | None:
    try:
        start = int(label["start"])
        end = int(label["end"])
    except (KeyError, TypeError, ValueError):
        return None
    start = max(0, min(start, text_length))
    end = max(0, min(end, text_length))
    if end <= start:
        return None
    return start, end


def merge_sentence_spans_for_labels(
    text: str,
    labels: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    spans = sentence_spans(text)
    if not spans:
        return []
    parent = list(range(len(spans)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for label in labels:
        span = _label_span(label, len(text))
        if span is None:
            continue
        label_start, label_end = span
        overlapping = [
            index
            for index, (start, end) in enumerate(spans)
            if label_start < end and label_end > start
        ]
        for first, second in zip(overlapping, overlapping[1:]):
            union(first, second)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(spans)):
        groups[find(index)].append(index)

    merged = [
        (spans[min(indices)][0], spans[max(indices)][1])
        for indices in groups.values()
    ]
    return sorted(merged)


def labels_overlapping(
    labels: list[dict[str, Any]],
    start: int,
    end: int,
    text_length: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in labels:
        span = _label_span(label, text_length)
        if span is None:
            continue
        label_start, label_end = span
        if label_start < end and label_end > start:
            result.append(label)
    return result


def parse_qa_passages(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("passages", "")
    text = str(value or "")
    matches = list(_PASSAGE_MARKER.finditer(text))
    if not matches:
        pieces = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        return pieces
    passages: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        passage = text[start:end].strip()
        if passage:
            passages.append(passage)
    return passages


def summary_windows(text: str, window_sentences: int = 3, stride_sentences: int = 2) -> list[str]:
    spans = sentence_spans(text)
    if not spans:
        return []
    sentences = [text[start:end].strip() for start, end in spans]
    if len(sentences) <= window_sentences:
        return [" ".join(sentences)]
    windows: list[str] = []
    for start in range(0, len(sentences), stride_sentences):
        window = " ".join(sentences[start : start + window_sentences]).strip()
        if window:
            windows.append(window)
        if start + window_sentences >= len(sentences):
            break
    return windows


def flatten_structured(value: Any, prefix: str = "") -> list[str]:
    rows: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_structured(child, child_prefix))
        return rows
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            rows.extend(flatten_structured(child, child_prefix))
        return rows
    rendered = "null" if value is None else str(value)
    rows.append(f"{prefix}: {rendered}" if prefix else rendered)
    return rows


def evidence_units_for_source(source_row: dict[str, Any]) -> list[str]:
    task_type = str(source_row.get("task_type", ""))
    source_info = source_row.get("source_info")
    if task_type == "QA":
        return parse_qa_passages(source_info)
    if task_type == "Summary":
        return summary_windows(str(source_info or ""))
    if task_type == "Data2txt":
        return flatten_structured(source_info)
    raise ValueError(f"task_type desconhecido: {task_type}")


def select_top_evidence(
    claim: str,
    units: list[str],
    top_k: int = 4,
    max_source_units: int = 128,
) -> tuple[list[str], list[bool]]:
    clean_units = [str(unit).strip() for unit in units if str(unit).strip()][:max_source_units]
    if not clean_units:
        return [""] * top_k, [False] * top_k
    if len(clean_units) > top_k:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r"(?u)\b\w+\b",
            )
            matrix = vectorizer.fit_transform([claim, *clean_units])
            similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
            order = np.argsort(-similarities, kind="stable")[:top_k]
            selected = [clean_units[int(index)] for index in order]
        except ValueError:
            selected = clean_units[:top_k]
    else:
        selected = clean_units[:top_k]
    mask = [True] * len(selected)
    while len(selected) < top_k:
        selected.append("")
        mask.append(False)
    return selected, mask


def response_claims(
    response_row: dict[str, Any],
    granularity: str,
    min_words: int,
) -> list[dict[str, Any]]:
    response = str(response_row.get("response", ""))
    labels = [label for label in response_row.get("labels", []) if isinstance(label, dict)]
    if granularity == "response":
        overlapping = labels
        return [
            {
                "claim_index": 0,
                "claim_start": 0,
                "claim_end": len(response),
                "claim": response.strip(),
                "label": bool(overlapping),
                "labels": overlapping,
            }
        ] if response.strip() else []
    if granularity != "claim":
        raise ValueError(f"Granularidade inválida: {granularity}")

    rows: list[dict[str, Any]] = []
    for claim_index, (start, end) in enumerate(merge_sentence_spans_for_labels(response, labels)):
        claim = response[start:end].strip()
        if len(re.findall(r"\w+", claim, flags=re.UNICODE)) < min_words:
            continue
        overlapping = labels_overlapping(labels, start, end, len(response))
        rows.append(
            {
                "claim_index": claim_index,
                "claim_start": start,
                "claim_end": end,
                "claim": claim,
                "label": bool(overlapping),
                "labels": overlapping,
            }
        )
    return rows


def label_independent_sentence_claims(
    response_row: dict[str, Any],
    min_words: int,
) -> list[dict[str, Any]]:
    response = str(response_row.get("response", ""))
    labels = [label for label in response_row.get("labels", []) if isinstance(label, dict)]
    sentence_units = sentence_spans(response)
    rows: list[dict[str, Any]] = []
    for claim_index, (start, end) in enumerate(sentence_units):
        claim = response[start:end]
        if len(re.findall(r"\w+", claim, flags=re.UNICODE)) < min_words:
            continue
        overlapping = labels_overlapping(labels, start, end, len(response))
        rows.append(
            {
                "claim_index": claim_index,
                "claim_start": start,
                "claim_end": end,
                "claim": claim,
                "label": bool(overlapping),
                "labels": overlapping,
            }
        )
    return rows


def _validation_sources(
    train_source_rows: list[dict[str, Any]],
    responses_by_source: dict[str, list[dict[str, Any]]],
    validation_fraction: float,
    seed: int,
) -> set[str]:
    source_ids = [str(row["source_id"]) for row in train_source_rows]
    strata = []
    for row in train_source_rows:
        source_id = str(row["source_id"])
        any_positive = any(bool(response.get("labels")) for response in responses_by_source[source_id])
        strata.append(f"{row.get('task_type')}|{int(any_positive)}")
    counts = Counter(strata)
    if any(count < 2 for count in counts.values()):
        strata = [str(row.get("task_type")) for row in train_source_rows]
    train_ids, validation_ids = train_test_split(
        source_ids,
        test_size=validation_fraction,
        random_state=seed,
        stratify=strata,
    )
    if set(train_ids) & set(validation_ids):
        raise AssertionError("Vazamento entre treino e validação por source_id.")
    return set(validation_ids)


def prepare_ragtruth(
    response_path: Path,
    source_path: Path,
    output_dir: Path,
    tasks: set[str],
    granularity: str = "claim",
    validation_fraction: float = 0.15,
    seed: int = 42,
    top_k: int = 4,
    min_words: int = 3,
    max_source_units: int = 128,
    drop_due_to_null_only: bool = False,
) -> dict[str, Any]:
    responses = read_jsonl(response_path)
    sources = read_jsonl(source_path)
    sources_by_id = {str(row["source_id"]): row for row in sources}
    if len(sources_by_id) != len(sources):
        raise ValueError("source_id duplicado em source_info.jsonl")

    responses_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        responses_by_source[str(response["source_id"])].append(response)
    for source_id, source_responses in responses_by_source.items():
        official_splits = {str(row.get("split")) for row in source_responses}
        if len(official_splits) > 1:
            raise ValueError(f"source_id {source_id} aparece em múltiplos splits oficiais: {official_splits}")

    filtered_sources = [row for row in sources if str(row.get("task_type")) in tasks]
    train_source_rows = [
        row
        for row in filtered_sources
        if any(str(response.get("split")) == "train" for response in responses_by_source[str(row["source_id"])])
    ]
    validation_ids = _validation_sources(
        train_source_rows,
        responses_by_source,
        validation_fraction,
        seed,
    )

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    skipped_quality = 0
    skipped_empty = 0
    invalid_span_text = 0

    for response in responses:
        source_id = str(response["source_id"])
        source_row = sources_by_id.get(source_id)
        if source_row is None or str(source_row.get("task_type")) not in tasks:
            continue
        if str(response.get("quality", "good")) != "good":
            skipped_quality += 1
            continue
        official_split = str(response.get("split"))
        split = "test" if official_split == "test" else ("validation" if source_id in validation_ids else "train")
        units = evidence_units_for_source(source_row)
        claims = response_claims(response, granularity=granularity, min_words=min_words)
        if not claims:
            skipped_empty += 1
            continue

        response_text = str(response.get("response", ""))
        for label in response.get("labels", []):
            if not isinstance(label, dict) or "text" not in label:
                continue
            span = _label_span(label, len(response_text))
            if span is None:
                invalid_span_text += 1
                continue
            start, end = span
            if str(label.get("text", "")).strip() and response_text[start:end] != str(label.get("text")):
                invalid_span_text += 1

        for claim_row in claims:
            evidence, evidence_mask = select_top_evidence(
                claim_row["claim"],
                units,
                top_k=top_k,
                max_source_units=max_source_units,
            )
            labels = claim_row.pop("labels")
            if drop_due_to_null_only and labels and all(bool(label.get("due_to_null", False)) for label in labels):
                continue
            split_rows[split].append(
                {
                    "example_id": f"{response['id']}:{claim_row['claim_index']}",
                    "source_id": source_id,
                    "response_id": str(response["id"]),
                    "claim_index": int(claim_row["claim_index"]),
                    "claim_start": int(claim_row["claim_start"]),
                    "claim_end": int(claim_row["claim_end"]),
                    "claim": claim_row["claim"],
                    "label": bool(claim_row["label"]),
                    "evidence": evidence,
                    "evidence_mask": evidence_mask,
                    "task_type": str(source_row.get("task_type")),
                    "source_name": str(source_row.get("source", "")),
                    "generator_model": str(response.get("model", "")),
                    "temperature": response.get("temperature"),
                    "granularity": granularity,
                    "label_types": sorted({str(label.get("label_type", "")) for label in labels}),
                    "implicit_true": any(bool(label.get("implicit_true", False)) for label in labels),
                    "due_to_null": any(bool(label.get("due_to_null", False)) for label in labels),
                }
            )

    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    audit_candidates = [dict(row, split=split) for split, rows in split_rows.items() for row in rows]
    audit_rng = random.Random(seed)
    audit_sample = audit_rng.sample(audit_candidates, k=min(100, len(audit_candidates)))
    audit_path = output_dir / "audit_sample.csv"
    audit_fields = [
        "split", "example_id", "source_id", "response_id", "task_type",
        "generator_model", "claim", "label", "label_types", "implicit_true",
        "due_to_null", "evidence_1", "evidence_2", "evidence_3", "evidence_4",
    ]
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for row in audit_sample:
            writer.writerow({
                "split": row["split"],
                "example_id": row["example_id"],
                "source_id": row["source_id"],
                "response_id": row["response_id"],
                "task_type": row["task_type"],
                "generator_model": row["generator_model"],
                "claim": row["claim"],
                "label": row["label"],
                "label_types": json.dumps(row["label_types"], ensure_ascii=False),
                "implicit_true": row["implicit_true"],
                "due_to_null": row["due_to_null"],
                **{f"evidence_{index + 1}": value for index, value in enumerate(row["evidence"])},
            })

    train_sources = {row["source_id"] for row in split_rows["train"]}
    validation_sources = {row["source_id"] for row in split_rows["validation"]}
    test_sources = {row["source_id"] for row in split_rows["test"]}
    if train_sources & validation_sources or train_sources & test_sources or validation_sources & test_sources:
        raise AssertionError("Vazamento de source_id entre splits.")

    stats: dict[str, Any] = {}
    for split, rows in split_rows.items():
        stats[split] = {
            "examples": len(rows),
            "positives": int(sum(bool(row["label"]) for row in rows)),
            "positive_rate": float(np.mean([bool(row["label"]) for row in rows])) if rows else 0.0,
            "sources": len({row["source_id"] for row in rows}),
            "tasks": dict(Counter(row["task_type"] for row in rows)),
        }

    manifest = {
        "dataset": "RAGTruth",
        "response_path": str(response_path.resolve()),
        "source_path": str(source_path.resolve()),
        "response_sha256": sha256_file(response_path),
        "source_sha256": sha256_file(source_path),
        "tasks": sorted(tasks),
        "granularity": granularity,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "top_k": top_k,
        "min_words": min_words,
        "max_source_units": max_source_units,
        "drop_due_to_null_only": drop_due_to_null_only,
        "filters": {"quality": "good"},
        "skipped_quality_responses": skipped_quality,
        "skipped_empty_responses": skipped_empty,
        "span_text_mismatches_or_invalid": invalid_span_text,
        "stats": stats,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
