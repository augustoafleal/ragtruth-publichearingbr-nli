from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALL_METRICS = ("heuristics", "cometkiwi", "nli_consistency")

THRESHOLD_CRITERIA = ("f1", "fpr10")
QUALITY_DIRECTIONS = ("higher_is_better", "lower_is_better")


def resolve_device(requested: str | None) -> str:
    import torch

    normalized = (requested or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError("CUDA foi solicitado, mas torch.cuda.is_available() é falso.")
        return normalized
    raise ValueError(f"device inválido: {requested!r} (use auto, cuda ou cpu)")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _signature_for(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"A seção {label} deve ser um mapa.")
    return value


@dataclass(frozen=True)
class AlignmentConfig:
    backend: str
    en_parquet: Path
    pt_parquet: Path
    output_dir: Path
    claim_column: str = "claim"
    chunk_columns: tuple[str, ...] = ("chunk_1", "chunk_2", "chunk_3", "chunk_4")
    evidence_mask_column: str = "evidence_mask"
    example_id_column: str = "example_id"
    source_id_column: str = "source_id"
    label_column: str = "label"
    split_column: str = "split"
    response_id_column: str = "response_id"
    expected_rows: int | None = None
    expected_chunking_signature: str | None = None
    expected_tokenizer_revision: str | None = None
    schema_version: str = "ragtruth-translation-quality-alignment-v1"

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend é obrigatório (por exemplo, nllb ou madlad).")
        if len(self.chunk_columns) != 4:
            raise ValueError("chunk_columns deve conter exatamente quatro colunas.")

    def recipe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "en_parquet_name": self.en_parquet.name,
            "pt_parquet_name": self.pt_parquet.name,
            "expected_rows": self.expected_rows,
            "expected_chunking_signature": self.expected_chunking_signature,
            "expected_tokenizer_revision": self.expected_tokenizer_revision,
            "claim_column": self.claim_column,
            "chunk_columns": list(self.chunk_columns),
        }

    @property
    def signature(self) -> str:
        return _signature_for(self.recipe(), 16)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.signature

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.recipe(),
            "en_parquet": str(self.en_parquet),
            "pt_parquet": str(self.pt_parquet),
            "output_dir": str(self.output_dir),
            "example_id_column": self.example_id_column,
            "source_id_column": self.source_id_column,
            "label_column": self.label_column,
            "split_column": self.split_column,
            "response_id_column": self.response_id_column,
            "evidence_mask_column": self.evidence_mask_column,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class ScoringConfig:
    backend: str
    aligned_parquet: Path
    output_dir: Path
    metrics: tuple[str, ...] = ("heuristics",)
    device: str = "auto"
    batch_size: int = 32
    sample_split: str | None = None
    sample_limit: int | None = None
    cometkiwi_model_id: str = "Unbabel/wmt22-cometkiwi-da"
    cometkiwi_model_revision: str | None = None
    nli_model_id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    nli_model_revision: str | None = None
    nli_max_length: int = 512
    entail_threshold: float = 0.5
    detector_truncation: bool = False
    detector_tokenizer_model_id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    detector_tokenizer_revision: str | None = None
    detector_max_length: int = 512
    schema_version: str = "ragtruth-translation-quality-scoring-v1"

    def __post_init__(self) -> None:
        unknown = [m for m in self.metrics if m not in ALL_METRICS]
        if unknown:
            raise ValueError(f"métricas desconhecidas: {unknown}; válidas: {list(ALL_METRICS)}")
        if not self.metrics:
            raise ValueError("metrics não pode ser vazio.")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser positivo.")
        if self.sample_limit is not None and self.sample_limit < 1:
            raise ValueError("sample_limit deve ser positivo.")
        if not 0.0 < self.entail_threshold < 1.0:
            raise ValueError("entail_threshold deve estar entre 0 e 1.")
        if "cometkiwi" in self.metrics and not self.cometkiwi_model_revision:
            raise ValueError("cometkiwi_model_revision é obrigatória quando cometkiwi está em metrics.")
        if "nli_consistency" in self.metrics and not self.nli_model_revision:
            raise ValueError("nli_model_revision é obrigatória quando nli_consistency está em metrics.")
        if self.detector_truncation and not self.detector_tokenizer_revision:
            raise ValueError(
                "detector_tokenizer_revision é obrigatória quando detector_truncation está habilitado."
            )

    def recipe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "metrics": list(self.metrics),
            "sample_split": self.sample_split,
            "sample_limit": self.sample_limit,
            "cometkiwi_model_id": self.cometkiwi_model_id if "cometkiwi" in self.metrics else None,
            "cometkiwi_model_revision": (
                self.cometkiwi_model_revision if "cometkiwi" in self.metrics else None
            ),
            "nli_model_id": self.nli_model_id if "nli_consistency" in self.metrics else None,
            "nli_model_revision": (
                self.nli_model_revision if "nli_consistency" in self.metrics else None
            ),
            "nli_max_length": self.nli_max_length,
            "entail_threshold": self.entail_threshold,
            "detector_truncation": self.detector_truncation,
            "detector_tokenizer_model_id": self.detector_tokenizer_model_id,
            "detector_tokenizer_revision": self.detector_tokenizer_revision,
            "detector_max_length": self.detector_max_length,
            "aligned_parquet_name": self.aligned_parquet.name,
        }

    @property
    def signature(self) -> str:
        return _signature_for(self.recipe(), 16)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.signature

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.recipe(),
            "aligned_parquet": str(self.aligned_parquet),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "batch_size": self.batch_size,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class LinkConfig:
    backend: str
    example_scores_parquet: Path
    predictions_parquet: Path
    output_dir: Path
    protocol_thresholds_json: Path | None = None
    threshold_criterion: str = "f1"
    expected_protocol_signature: str | None = None
    quality_signal: str = "nli_abs_delta_max"
    quality_direction: str = "lower_is_better"
    num_quality_buckets: int = 4
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    sample_limit: int | None = None
    example_id_column: str = "example_id"
    source_id_column: str = "source_id"
    label_column: str = "label"
    score_column: str = "score"
    schema_version: str = "ragtruth-translation-quality-link-v1"

    def __post_init__(self) -> None:
        if self.threshold_criterion not in THRESHOLD_CRITERIA:
            raise ValueError(f"threshold_criterion deve ser um de {THRESHOLD_CRITERIA}.")
        if self.quality_direction not in QUALITY_DIRECTIONS:
            raise ValueError(f"quality_direction deve ser um de {QUALITY_DIRECTIONS}.")
        if self.num_quality_buckets < 2:
            raise ValueError("num_quality_buckets deve ser >= 2.")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples deve ser positivo.")
        if self.sample_limit is not None and self.sample_limit < 1:
            raise ValueError("sample_limit deve ser positivo.")

    @property
    def thresholds_json(self) -> Path:
        if self.protocol_thresholds_json is not None:
            return self.protocol_thresholds_json
        return self.predictions_parquet.parent / "thresholds_applied.json"

    def recipe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "quality_signal": self.quality_signal,
            "quality_direction": self.quality_direction,
            "num_quality_buckets": self.num_quality_buckets,
            "threshold_criterion": self.threshold_criterion,
            "expected_protocol_signature": self.expected_protocol_signature,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "sample_limit": self.sample_limit,
            "example_scores_name": self.example_scores_parquet.name,
            "predictions_name": self.predictions_parquet.name,
        }

    @property
    def signature(self) -> str:
        return _signature_for(self.recipe(), 16)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.signature

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.recipe(),
            "example_scores_parquet": str(self.example_scores_parquet),
            "predictions_parquet": str(self.predictions_parquet),
            "protocol_thresholds_json": str(self.thresholds_json),
            "output_dir": str(self.output_dir),
            "example_id_column": self.example_id_column,
            "source_id_column": self.source_id_column,
            "label_column": self.label_column,
            "score_column": self.score_column,
            "signature": self.signature,
        }


def alignment_from_mapping(raw: dict[str, Any], base_dir: Path) -> AlignmentConfig:
    backend = str(raw.get("backend", "")).strip()
    section = _as_mapping(raw.get("alignment", {}), "alignment")
    for key in ("en_parquet", "pt_parquet"):
        if not section.get(key):
            raise ValueError(f"alignment.{key} é obrigatório.")
    return AlignmentConfig(
        backend=backend,
        en_parquet=_resolve_path(section["en_parquet"], base_dir),
        pt_parquet=_resolve_path(section["pt_parquet"], base_dir),
        output_dir=_resolve_path(
            section.get("output_dir", f"results/translation_quality/{backend}/alignment"), base_dir
        ),
        claim_column=str(section.get("claim_column", "claim")),
        chunk_columns=tuple(section.get("chunk_columns", ["chunk_1", "chunk_2", "chunk_3", "chunk_4"])),
        evidence_mask_column=str(section.get("evidence_mask_column", "evidence_mask")),
        example_id_column=str(section.get("example_id_column", "example_id")),
        source_id_column=str(section.get("source_id_column", "source_id")),
        label_column=str(section.get("label_column", "label")),
        split_column=str(section.get("split_column", "split")),
        response_id_column=str(section.get("response_id_column", "response_id")),
        expected_rows=(int(section["expected_rows"]) if section.get("expected_rows") is not None else None),
        expected_chunking_signature=(
            str(section["expected_chunking_signature"])
            if section.get("expected_chunking_signature")
            else None
        ),
        expected_tokenizer_revision=(
            str(section["expected_tokenizer_revision"])
            if section.get("expected_tokenizer_revision")
            else None
        ),
    )


def scoring_from_mapping(raw: dict[str, Any], base_dir: Path) -> ScoringConfig:
    backend = str(raw.get("backend", "")).strip()
    section = _as_mapping(raw.get("scoring", {}), "scoring")
    aligned = section.get("aligned_parquet")
    if aligned:
        aligned_parquet = _resolve_path(aligned, base_dir)
    else:
        alignment = alignment_from_mapping(raw, base_dir)
        aligned_parquet = alignment.run_dir / "aligned.parquet"
    return ScoringConfig(
        backend=backend,
        aligned_parquet=aligned_parquet,
        output_dir=_resolve_path(
            section.get("output_dir", f"results/translation_quality/{backend}/scoring"), base_dir
        ),
        metrics=tuple(section.get("metrics", ["heuristics"])),
        device=str(section.get("device", "auto")),
        batch_size=int(section.get("batch_size", 32)),
        sample_split=(str(section["sample_split"]) if section.get("sample_split") else None),
        sample_limit=(int(section["sample_limit"]) if section.get("sample_limit") is not None else None),
        cometkiwi_model_id=str(section.get("cometkiwi_model_id", "Unbabel/wmt22-cometkiwi-da")),
        cometkiwi_model_revision=(
            str(section["cometkiwi_model_revision"])
            if section.get("cometkiwi_model_revision")
            else None
        ),
        nli_model_id=str(
            section.get("nli_model_id", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
        ),
        nli_model_revision=(
            str(section["nli_model_revision"]) if section.get("nli_model_revision") else None
        ),
        nli_max_length=int(section.get("nli_max_length", 512)),
        entail_threshold=float(section.get("entail_threshold", 0.5)),
        detector_truncation=bool(section.get("detector_truncation", False)),
        detector_tokenizer_model_id=str(
            section.get(
                "detector_tokenizer_model_id",
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            )
        ),
        detector_tokenizer_revision=(
            str(section["detector_tokenizer_revision"])
            if section.get("detector_tokenizer_revision")
            else None
        ),
        detector_max_length=int(section.get("detector_max_length", 512)),
    )


def link_from_mapping(raw: dict[str, Any], base_dir: Path) -> LinkConfig:
    backend = str(raw.get("backend", "")).strip()
    section = _as_mapping(raw.get("link", {}), "link")
    if not section.get("predictions_parquet"):
        raise ValueError("link.predictions_parquet é obrigatório para o Stage 2.")
    scores = section.get("example_scores_parquet")
    if scores:
        example_scores = _resolve_path(scores, base_dir)
    else:
        scoring = scoring_from_mapping(raw, base_dir)
        example_scores = scoring.run_dir / "example_scores.parquet"
    thresholds = section.get("protocol_thresholds_json")
    return LinkConfig(
        backend=backend,
        example_scores_parquet=example_scores,
        predictions_parquet=_resolve_path(section["predictions_parquet"], base_dir),
        output_dir=_resolve_path(
            section.get("output_dir", f"results/translation_quality/{backend}/link"), base_dir
        ),
        protocol_thresholds_json=(_resolve_path(thresholds, base_dir) if thresholds else None),
        threshold_criterion=str(section.get("threshold_criterion", "f1")),
        expected_protocol_signature=(
            str(section["expected_protocol_signature"])
            if section.get("expected_protocol_signature")
            else None
        ),
        quality_signal=str(section.get("quality_signal", "nli_abs_delta_max")),
        quality_direction=str(section.get("quality_direction", "lower_is_better")),
        num_quality_buckets=int(section.get("num_quality_buckets", 4)),
        bootstrap_samples=int(section.get("bootstrap_samples", 1000)),
        bootstrap_seed=int(section.get("bootstrap_seed", 42)),
        sample_limit=(int(section["sample_limit"]) if section.get("sample_limit") is not None else None),
        example_id_column=str(section.get("example_id_column", "example_id")),
        source_id_column=str(section.get("source_id_column", "source_id")),
        label_column=str(section.get("label_column", "label")),
        score_column=str(section.get("score_column", "score")),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuração inválida: {path}")
    return raw


def load_alignment_config(path: Path) -> AlignmentConfig:
    path = Path(path).resolve()
    return alignment_from_mapping(_load_yaml(path), base_dir=path.parent)


def load_scoring_config(path: Path) -> ScoringConfig:
    path = Path(path).resolve()
    return scoring_from_mapping(_load_yaml(path), base_dir=path.parent)


def load_link_config(path: Path) -> LinkConfig:
    path = Path(path).resolve()
    return link_from_mapping(_load_yaml(path), base_dir=path.parent)
