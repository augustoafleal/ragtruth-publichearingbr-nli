from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


DEFAULT_MIN_LENGTH_RATIO = 0.40
DEFAULT_MAX_LENGTH_RATIO = 2.50
DEFAULT_REPETITION_MIN_RUN = 4
_TEXT_RE = re.compile(r"\S+")


def text_value(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", text_value(value)).strip()


def _tokens(value: Any) -> list[str]:
    return _TEXT_RE.findall(normalize_text(value).lower())


def _repetition_runs(value: Any) -> dict[int, int]:
    tokens = _tokens(value)
    if not tokens:
        return {1: 0, 2: 0, 3: 0}

    result: dict[int, int] = {}
    for size in (1, 2, 3):
        best = 1
        for offset in range(size):
            current = 0
            previous: tuple[str, ...] | None = None
            for start in range(offset, len(tokens) - size + 1, size):
                unit = tuple(tokens[start : start + size])
                current = current + 1 if unit == previous else 1
                best = max(best, current)
                previous = unit
        result[size] = best
    return result


def compute_length_metrics(
    source_text: Any,
    translated_text: Any,
    *,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
) -> dict[str, Any]:
    source = text_value(source_text)
    translated = text_value(translated_text)
    source_normalized = normalize_text(source)
    translated_normalized = normalize_text(translated)
    source_chars = len(source)
    translated_chars = len(translated)
    length_ratio = translated_chars / source_chars if source_chars else None
    return {
        "source_normalized": source_normalized,
        "translated_normalized": translated_normalized,
        "source_chars": source_chars,
        "translated_chars": translated_chars,
        "source_words": len(source_normalized.split()),
        "translated_words": len(translated_normalized.split()),
        "length_ratio": length_ratio,
        "empty_translation": bool(source_normalized and not translated_normalized),
        "identical_to_source": source_normalized == translated_normalized,
        "low_ratio": length_ratio is not None and length_ratio < min_length_ratio,
        "high_ratio": length_ratio is not None and length_ratio > max_length_ratio,
    }


def compute_repetition_metrics(
    source_text: Any,
    translated_text: Any,
    *,
    repetition_min_run: int = DEFAULT_REPETITION_MIN_RUN,
) -> dict[str, Any]:
    source_run = max(_repetition_runs(source_text).values())
    translated_run = max(_repetition_runs(translated_text).values())
    source_has_repetition = source_run >= 2
    translation_has_repetition = translated_run >= 2
    repetition_excess = translated_run - source_run
    translation_added = translation_has_repetition and repetition_excess > 0
    if not translation_added:
        severity = "none"
    elif translated_run >= repetition_min_run and repetition_excess >= 2:
        severity = "high"
    elif translated_run >= 3:
        severity = "medium"
    else:
        severity = "low"
    return {
        "source_has_repetition": source_has_repetition,
        "translation_has_repetition": translation_has_repetition,
        "source_repetition_metric": source_run,
        "translation_repetition_metric": translated_run,
        "repetition_excess": repetition_excess,
        "repetition_severity": severity,
        "translation_added_repetition": translation_added,
        "high_confidence_repetition": severity == "high",
    }


def has_unexpected_control(value: Any) -> bool:
    return any(
        (ord(char) < 32 and char not in "\n\t") or 0x7F <= ord(char) <= 0x9F
        for char in text_value(value)
    )


def classify_translation_pair(
    source_text: Any,
    translated_text: Any,
    *,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    repetition_min_run: int = DEFAULT_REPETITION_MIN_RUN,
) -> dict[str, Any]:
    metrics = compute_length_metrics(
        source_text,
        translated_text,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
    )
    metrics.update(
        compute_repetition_metrics(
            source_text,
            translated_text,
            repetition_min_run=repetition_min_run,
        )
    )
    metrics["control_character_issue"] = has_unexpected_control(translated_text)
    metrics["exclusion_candidate"] = bool(
        metrics["low_ratio"]
        or metrics["high_ratio"]
        or metrics["high_confidence_repetition"]
    )
    return metrics


def stable_pair_id(source_normalized: str, translated_normalized: str) -> str:
    payload = json.dumps(
        [source_normalized, translated_normalized],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_exclusion_candidates(
    pairs: Iterable[tuple[Any, Any]],
    *,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    repetition_min_run: int = DEFAULT_REPETITION_MIN_RUN,
) -> dict[tuple[str, str], dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for source_text, translated_text in pairs:
        metrics = classify_translation_pair(
            source_text,
            translated_text,
            min_length_ratio=min_length_ratio,
            max_length_ratio=max_length_ratio,
            repetition_min_run=repetition_min_run,
        )
        identity = (metrics["source_normalized"], metrics["translated_normalized"])
        if metrics["exclusion_candidate"]:
            candidates.setdefault(identity, metrics)
    return candidates


def structural_mismatch_counts(source: dict[str, Any], translated: dict[str, Any]) -> dict[str, int]:
    counts = {
        "id_mismatches": 0,
        "label_mismatches": 0,
        "mask_mismatches": 0,
        "evidence_slot_mismatches": 0,
        "metadata_mismatches": 0,
        "schema_mismatches": 0,
        "masked_evidence_changes": 0,
    }
    if set(source) != set(translated):
        counts["schema_mismatches"] = 1
    if (source.get("example_id"), source.get("source_id")) != (
        translated.get("example_id"),
        translated.get("source_id"),
    ):
        counts["id_mismatches"] = 1
    if source.get("label") != translated.get("label"):
        counts["label_mismatches"] = 1
    if source.get("evidence_mask") != translated.get("evidence_mask"):
        counts["mask_mismatches"] = 1

    allowed = {"claim", "evidence"}
    separately_reported = {"example_id", "source_id", "label", "evidence_mask"} | allowed
    metadata_keys = (set(source) | set(translated)) - separately_reported
    if any(source.get(key) != translated.get(key) for key in metadata_keys):
        counts["metadata_mismatches"] = 1

    source_evidence = source.get("evidence", [])
    translated_evidence = translated.get("evidence", [])
    if len(source_evidence) != len(translated_evidence):
        counts["evidence_slot_mismatches"] = 1
    mask = source.get("evidence_mask", [])
    if any(
        index >= len(translated_evidence)
        or (index < len(source_evidence) and source_evidence[index] != translated_evidence[index])
        for index, is_valid in enumerate(mask)
        if is_valid is False
    ):
        counts["masked_evidence_changes"] = 1
    return counts
