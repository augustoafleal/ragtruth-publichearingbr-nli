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
    source_text: str
    target_text: str
    claim_source: str
    claim_target: str


def _sample(frame: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    if config.sample_split:
        frame = frame[frame["split"] == config.sample_split]
    frame = frame.sort_values("example_id").reset_index(drop=True)
    if config.sample_limit is not None:
        frame = frame.head(config.sample_limit).reset_index(drop=True)
    return frame


def _pair_columns(frame: pd.DataFrame, stem: str, config: ScoringConfig) -> tuple[str, str]:
    generic = (f"{stem}_source", f"{stem}_target")
    if all(column in frame.columns for column in generic):
        return generic
    language_columns = (f"{stem}_{config.source_language}", f"{stem}_{config.target_language}")
    if all(column in frame.columns for column in language_columns):
        return language_columns
    if stem == "claim" and {"claim_en", "claim_pt"}.issubset(frame.columns):
        return ("claim_en", "claim_pt")
    if stem.startswith("chunk_"):
        slot = stem.split("_", 1)[1]
        if {f"chunk_{slot}_en", f"chunk_{slot}_pt"}.issubset(frame.columns):
            return (f"chunk_{slot}_en", f"chunk_{slot}_pt")
    raise ValueError(f"Aligned artifact lacks source/target columns for {stem}.")


def _build_segments(frame: pd.DataFrame, config: ScoringConfig) -> list[Segment]:
    segments: list[Segment] = []
    claim_source_column, claim_target_column = _pair_columns(frame, "claim", config)
    for _, row in frame.iterrows():
        claim_source = "" if row.get(claim_source_column) is None else str(row.get(claim_source_column))
        claim_target = "" if row.get(claim_target_column) is None else str(row.get(claim_target_column))
        common = dict(
            example_id=str(row["example_id"]),
            source_id=str(row.get("source_id")),
            split=row.get("split"),
            label=(bool(row["label"]) if row.get("label") is not None else None),
            claim_source=claim_source,
            claim_target=claim_target,
        )
        segments.append(Segment(role="claim", slot=0, source_text=claim_source, target_text=claim_target, **common))
        for slot in range(1, 5):
            if bool(row.get(f"chunk_{slot}_valid", False)):
                source_column, target_column = _pair_columns(frame, f"chunk_{slot}", config)
                segments.append(
                    Segment(
                        role="chunk",
                        slot=slot,
                        source_text=str(row.get(source_column) or ""),
                        target_text=str(row.get(target_column) or ""),
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
            "source_text": s.source_text,
            "target_text": s.target_text,
            f"source_{config.source_language}": s.source_text,
            f"translated_{config.target_language}": s.target_text,
            # Historical column names remain available for the EN→PT RAGTruth
            # outputs; their values are always language-labelled, not positional.
            "source_en": s.source_text if config.source_language == "en" else s.target_text,
            "translated_pt": s.target_text if config.target_language == "pt" else s.source_text,
        }
        for s in segments
    ]

    pairs = [(s.source_text, s.target_text) for s in segments]
    for scorer in (scorers.heuristics, scorers.cometkiwi):
        if scorer is None:
            continue
        scored = scorer.score_pairs(pairs)
        for record, values in zip(records, scored):
            record.update(values)

    chunk_idx = [i for i, s in enumerate(segments) if s.role == "chunk"]
    if scorers.nli is not None or scorers.detector_tokenizer is not None:
        source_pairs = [(segments[i].source_text, segments[i].claim_source) for i in chunk_idx]
        target_pairs = [(segments[i].target_text, segments[i].claim_target) for i in chunk_idx]

        if scorers.nli is not None:
            probs_source = scorers.nli.entailment_probs(source_pairs)
            probs_target = scorers.nli.entailment_probs(target_pairs)
            thr = config.entail_threshold
            for position, i in enumerate(chunk_idx):
                p_source = float(probs_source[position])
                p_target = float(probs_target[position])
                delta = p_target - p_source
                records[i].update(
                    {
                        "nli_entail_source": p_source,
                        "nli_entail_target": p_target,
                        "nli_entail_en": p_source if config.source_language == "en" else p_target,
                        "nli_entail_pt": p_source if config.source_language == "pt" else p_target,
                        "nli_entail_delta": delta,
                        "nli_abs_delta": abs(delta),
                        "nli_entail_abs_delta": abs(delta),
                        "nli_label_agree": bool((p_source >= thr) == (p_target >= thr)),
                    }
                )

        if scorers.detector_tokenizer is not None:
            truncations = detector_truncation_flags(
                scorers.detector_tokenizer, source_pairs, target_pairs, config.detector_max_length
            )
            for position, i in enumerate(chunk_idx):
                truncation = dict(truncations[position])
                source_tokens = truncation["detector_pair_tokens_en"]
                target_tokens = truncation["detector_pair_tokens_pt"]
                source_truncated = truncation["detector_truncated_en"]
                target_truncated = truncation["detector_truncated_pt"]
                records[i].update(
                    {
                        "detector_pair_tokens_source": source_tokens,
                        "detector_pair_tokens_target": target_tokens,
                        "detector_truncated_source": source_truncated,
                        "detector_truncated_target": target_truncated,
                        "detector_pair_tokens_en": target_tokens
                        if config.source_language == "pt"
                        else source_tokens,
                        "detector_pair_tokens_pt": source_tokens
                        if config.source_language == "pt"
                        else target_tokens,
                        "detector_truncated_en": target_truncated
                        if config.source_language == "pt"
                        else source_truncated,
                        "detector_truncated_pt": source_truncated
                        if config.source_language == "pt"
                        else target_truncated,
                        "detector_truncation_introduced": bool(
                            target_tokens > config.detector_max_length
                            and source_tokens <= config.detector_max_length
                        ),
                    }
                )
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
            record["nli_abs_delta"] = record["nli_abs_delta_mean"]

        if has_trunc:
            record["any_truncation_introduced"] = bool(
                chunk_rows["detector_truncation_introduced"].fillna(False).any()
            )
            pt_tokens = chunk_rows["detector_pair_tokens_pt"].dropna()
            record["max_detector_pair_tokens_pt"] = int(pt_tokens.max()) if len(pt_tokens) else 0
            target_tokens = chunk_rows["detector_pair_tokens_target"].dropna()
            record["max_detector_pair_tokens_target"] = (
                int(target_tokens.max()) if len(target_tokens) else 0
            )

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
    segments = _build_segments(frame, config)

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
