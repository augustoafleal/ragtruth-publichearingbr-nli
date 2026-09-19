from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..io_utils import sha256_file
from .config import ScoringConfig, resolve_device
from .metrics import (
    CometKiwiScorer,
    HeuristicScorer,
    NLIEntailmentScorer,
    detector_truncation_flags,
)
from .storage import write_json_atomic, write_parquet_atomic

MANIFEST_SCHEMA = "ragtruth-translation-quality-scoring-manifest-v1"


@dataclass
class ScorerBundle:
    heuristics: Any | None = None
    cometkiwi: Any | None = None
    nli: Any | None = None
    detector_tokenizer: Any | None = None


def build_scorers(config: ScoringConfig) -> ScorerBundle:
    bundle = ScorerBundle()
    if "heuristics" in config.metrics:
        bundle.heuristics = HeuristicScorer()
    if "cometkiwi" in config.metrics:
        bundle.cometkiwi = CometKiwiScorer.load(
            config.cometkiwi_model_id,
            config.cometkiwi_model_revision,
            config.device,
            config.batch_size,
        )
    if "nli_consistency" in config.metrics:
        bundle.nli = NLIEntailmentScorer.load(
            config.nli_model_id,
            config.nli_model_revision,
            config.device,
            config.batch_size,
            config.nli_max_length,
        )
    if config.detector_truncation:
        bundle.detector_tokenizer = _load_detector_tokenizer(config)
    return bundle

def _load_detector_tokenizer(config: ScoringConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.detector_tokenizer_model_id, revision=config.detector_tokenizer_revision
    )


@dataclass
class Segment:
    example_id: str
    source_id: str
    split: Any
    label: Any
    role: str
    slot: int
    source_en: str
    translated_pt: str
    claim_en: str
    claim_pt: str


def _sample(frame: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    if config.sample_split:
        frame = frame[frame["split"] == config.sample_split]
    frame = frame.sort_values("example_id").reset_index(drop=True)
    if config.sample_limit is not None:
        frame = frame.head(config.sample_limit).reset_index(drop=True)
    return frame


def _build_segments(frame: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for _, row in frame.iterrows():
        claim_en = "" if row.get("claim_en") is None else str(row.get("claim_en"))
        claim_pt = "" if row.get("claim_pt") is None else str(row.get("claim_pt"))
        common = dict(
            example_id=str(row["example_id"]),
            source_id=str(row.get("source_id")),
            split=row.get("split"),
            label=(bool(row["label"]) if row.get("label") is not None else None),
            claim_en=claim_en,
            claim_pt=claim_pt,
        )
        segments.append(
            Segment(role="claim", slot=0, source_en=claim_en, translated_pt=claim_pt, **common)
        )
        for slot in range(1, 5):
            if bool(row.get(f"chunk_{slot}_valid", False)):
                segments.append(
                    Segment(
                        role="chunk",
                        slot=slot,
                        source_en=str(row.get(f"chunk_{slot}_en") or ""),
                        translated_pt=str(row.get(f"chunk_{slot}_pt") or ""),
                        **common,
                    )
                )
    return segments


def score_segments(
    config: ScoringConfig,
    segments: list[Segment],
    scorers: ScorerBundle,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = [
        {
            "example_id": s.example_id,
            "source_id": s.source_id,
            "split": s.split,
            "label": s.label,
            "role": s.role,
            "slot": s.slot,
            "source_en": s.source_en,
            "translated_pt": s.translated_pt,
        }
        for s in segments
    ]

    pairs = [(s.source_en, s.translated_pt) for s in segments]
    for scorer in (scorers.heuristics, scorers.cometkiwi):
        if scorer is None:
            continue
        scored = scorer.score_pairs(pairs)
        for record, values in zip(records, scored):
            record.update(values)

    chunk_idx = [i for i, s in enumerate(segments) if s.role == "chunk"]
    if scorers.nli is not None or scorers.detector_tokenizer is not None:
        en_pairs = [(segments[i].source_en, segments[i].claim_en) for i in chunk_idx]
        pt_pairs = [(segments[i].translated_pt, segments[i].claim_pt) for i in chunk_idx]

        if scorers.nli is not None:
            probs_en = scorers.nli.entailment_probs(en_pairs)
            probs_pt = scorers.nli.entailment_probs(pt_pairs)
            thr = config.entail_threshold
            for position, i in enumerate(chunk_idx):
                p_en = float(probs_en[position])
                p_pt = float(probs_pt[position])
                delta = p_pt - p_en
                records[i].update(
                    {
                        "nli_entail_en": p_en,
                        "nli_entail_pt": p_pt,
                        "nli_entail_delta": delta,
                        "nli_entail_abs_delta": abs(delta),
                        "nli_label_agree": bool((p_en >= thr) == (p_pt >= thr)),
                    }
                )

        if scorers.detector_tokenizer is not None:
            truncations = detector_truncation_flags(
                scorers.detector_tokenizer, en_pairs, pt_pairs, config.detector_max_length
            )
            for position, i in enumerate(chunk_idx):
                records[i].update(truncations[position])
    return pd.DataFrame.from_records(records)


def _isnan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def aggregate_examples(segment_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    has_heur = "heur_exclusion_candidate" in segment_frame.columns
    has_comet = "cometkiwi" in segment_frame.columns
    has_nli = "nli_entail_abs_delta" in segment_frame.columns
    has_trunc = "detector_truncation_introduced" in segment_frame.columns

    for example_id, group in segment_frame.groupby("example_id", sort=True):
        claim_rows = group[group["role"] == "claim"]
        chunk_rows = group[group["role"] == "chunk"]
        record: dict[str, Any] = {
            "example_id": example_id,
            "source_id": group["source_id"].iloc[0],
            "split": group["split"].iloc[0],
            "label": group["label"].iloc[0],
            "num_valid_chunks": int(len(chunk_rows)),
        }

        if has_heur:
            record["any_exclusion_candidate"] = bool(group["heur_exclusion_candidate"].any())
            record["any_high_ratio"] = bool(group.get("heur_high_ratio", pd.Series(dtype=bool)).any())
            record["any_low_ratio"] = bool(group.get("heur_low_ratio", pd.Series(dtype=bool)).any())
            record["any_high_repetition"] = bool(
                (group["heur_repetition_severity"] == "high").any()
            )

        if has_comet:
            claim_values = claim_rows["cometkiwi"].dropna()
            chunk_values = chunk_rows["cometkiwi"].dropna()
            record["cometkiwi_claim"] = float(claim_values.iloc[0]) if len(claim_values) else np.nan
            record["cometkiwi_chunk_mean"] = (
                float(chunk_values.mean()) if len(chunk_values) else np.nan
            )
            record["cometkiwi_chunk_min"] = float(chunk_values.min()) if len(chunk_values) else np.nan

        if has_nli:
            abs_delta = chunk_rows["nli_entail_abs_delta"].dropna()
            agree = chunk_rows["nli_label_agree"].dropna()
            record["nli_abs_delta_mean"] = float(abs_delta.mean()) if len(abs_delta) else np.nan
            record["nli_abs_delta_max"] = float(abs_delta.max()) if len(abs_delta) else np.nan
            record["nli_label_flips"] = int((~agree.astype(bool)).sum()) if len(agree) else 0
            record["nli_any_flip"] = bool(record["nli_label_flips"] > 0)

        if has_trunc:
            record["any_truncation_introduced"] = bool(
                chunk_rows["detector_truncation_introduced"].fillna(False).any()
            )
            pt_tokens = chunk_rows["detector_pair_tokens_pt"].dropna()
            record["max_detector_pair_tokens_pt"] = int(pt_tokens.max()) if len(pt_tokens) else 0

        records.append(record)

    return pd.DataFrame.from_records(records)


@dataclass
class ScoringResult:
    config: ScoringConfig
    segment_frame: pd.DataFrame
    example_frame: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def _summarize(example_frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"examples": int(len(example_frame))}
    for column in [
        "cometkiwi_claim",
        "cometkiwi_chunk_mean",
        "cometkiwi_chunk_min",
        "nli_abs_delta_mean",
        "nli_abs_delta_max",
        "max_detector_pair_tokens_pt",
    ]:
        if column in example_frame.columns:
            series = example_frame[column].dropna()
            if len(series):
                summary[column] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "p90": float(series.quantile(0.90)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
    for column in [
        "nli_any_flip",
        "any_exclusion_candidate",
        "any_truncation_introduced",
    ]:
        if column in example_frame.columns:
            summary[column + "_rate"] = float(example_frame[column].fillna(False).mean())
    return summary


def score_translation_quality(
    config: ScoringConfig,
    *,
    scorers: ScorerBundle | None = None,
    write: bool = True,
) -> ScoringResult:
    frame = pd.read_parquet(config.aligned_parquet)
    frame = _sample(frame, config)
    segments = _build_segments(frame)

    if scorers is None:
        scorers = build_scorers(config)

    segment_frame = score_segments(config, segments, scorers)
    example_frame = aggregate_examples(segment_frame)
    summary = _summarize(example_frame)
    result = ScoringResult(config, segment_frame, example_frame, summary)

    if write:
        run_dir = config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(run_dir / "segment_scores.parquet", segment_frame)
        write_parquet_atomic(run_dir / "example_scores.parquet", example_frame)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "signature": config.signature,
            "backend": config.backend,
            "config": config.to_dict(),
            "device": resolve_device(config.device),
            "inputs": {"aligned_parquet_sha256": sha256_file(config.aligned_parquet)},
            "summary": summary,
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
        write_json_atomic(run_dir / "resolved_config.json", config.to_dict())
        write_json_atomic(run_dir / "quality_summary.json", summary)
    return result
