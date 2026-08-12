
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSettings:
    response_path: Path
    source_path: Path
    revision: str | None = None
    task: str = "QA"
    splits: tuple[str, ...] = ("train", "test")
    quality: str = "good"
    granularity: str = "claim"
    min_words: int = 3
    boundary_strategy: str = "label_dependent_sentences"
    segmentation_version: str = "sentence-spans-v1"
    overlap_rule: str = "half_open_character_intersection-v1"
    duplicate_policy: str = "keep_all"


@dataclass(frozen=True)
class RetrieverSettings:
    model_id: str = "intfloat/multilingual-e5-base"
    revision: str | None = None
    tokenizer_model_id: str = "intfloat/multilingual-e5-base"
    tokenizer_revision: str | None = None
    encoder_backend: str = "transformers"
    embedding_dim: int = 768
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 64
    batch_size: int = 64
    device: str = "auto"
    dtype: str = "float32"
    prefix_query: str = "query: "
    prefix_passage: str = "passage: "
    strategy: str = "topk_cosine"
    normalize_whitespace: bool = True
    max_length: int | None = None


@dataclass(frozen=True)
class Top4Config:
    run_name: str
    output_root: Path
    cache_root: Path
    dataset: DatasetSettings
    retriever: RetrieverSettings = field(default_factory=RetrieverSettings)
    top_k: int = 4
    sample_size: int = 100
    audit_seed: int = 42
    schema_version: str = "ragtruth-qa-top4-v1"
    legacy_run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.dataset.task != "QA":
            raise ValueError("Esta etapa suporta apenas task=QA.")
        if not self.dataset.splits:
            raise ValueError("dataset.splits não pode ser vazio.")
        if any(split not in {"train", "test"} for split in self.dataset.splits):
            raise ValueError("Os splits oficiais aceitos são train e test.")
        if self.dataset.granularity not in {"claim", "response"}:
            raise ValueError("dataset.granularity deve ser claim ou response.")
        if self.dataset.min_words < 1:
            raise ValueError("dataset.min_words deve ser positivo.")
        if self.dataset.boundary_strategy not in {"label_dependent_sentences", "label_independent_sentences"}:
            raise ValueError("boundary_strategy inválida.")
        if self.dataset.duplicate_policy not in {"keep_all", "deduplicate_within_response"}:
            raise ValueError("duplicate_policy inválida.")
        if self.top_k != 4:
            raise ValueError("O contrato desta etapa exige exatamente top_k=4.")
        if self.retriever.chunk_size_tokens < 1:
            raise ValueError("chunk_size_tokens deve ser positivo.")
        if not 0 <= self.retriever.chunk_overlap_tokens < self.retriever.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens deve ser >=0 e menor que chunk_size_tokens.")
        if self.retriever.strategy != "topk_cosine":
            raise ValueError("A estratégia oficial desta etapa é topk_cosine.")
        if self.retriever.encoder_backend not in {"transformers", "mock"}:
            raise ValueError("encoder_backend deve ser transformers ou mock.")
        if self.retriever.dtype not in {"float32", "float16", "bfloat16", "auto"}:
            raise ValueError("dtype inválido.")
        if self.retriever.batch_size < 1:
            raise ValueError("batch_size deve ser positivo.")
        if self.retriever.encoder_backend == "transformers":
            if not self.retriever.revision or not self.retriever.tokenizer_revision:
                raise ValueError(
                    "revision e tokenizer_revision são obrigatórias para o encoder transformers; "
                    "use um SHA imutável, não main."
                )
            if self.retriever.revision != self.retriever.tokenizer_revision:
                raise ValueError("revision e tokenizer_revision devem apontar para o mesmo commit.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "output_root": str(self.output_root),
            "cache_root": str(self.cache_root),
            "top_k": self.top_k,
            "sample_size": self.sample_size,
            "audit_seed": self.audit_seed,
            "legacy_run_dir": str(self.legacy_run_dir) if self.legacy_run_dir else None,
            "dataset": {
                "response_path": str(self.dataset.response_path),
                "source_path": str(self.dataset.source_path),
                "revision": self.dataset.revision,
                "task": self.dataset.task,
                "splits": list(self.dataset.splits),
                "quality": self.dataset.quality,
                "granularity": self.dataset.granularity,
                "min_words": self.dataset.min_words,
                "boundary_strategy": self.dataset.boundary_strategy,
                "segmentation_version": self.dataset.segmentation_version,
                "overlap_rule": self.dataset.overlap_rule,
                "duplicate_policy": self.dataset.duplicate_policy,
            },
            "retriever": {
                "model_id": self.retriever.model_id,
                "revision": self.retriever.revision,
                "tokenizer_model_id": self.retriever.tokenizer_model_id,
                "tokenizer_revision": self.retriever.tokenizer_revision,
                "encoder_backend": self.retriever.encoder_backend,
                "embedding_dim": self.retriever.embedding_dim,
                "chunk_size_tokens": self.retriever.chunk_size_tokens,
                "chunk_overlap_tokens": self.retriever.chunk_overlap_tokens,
                "batch_size": self.retriever.batch_size,
                "device": self.retriever.device,
                "dtype": self.retriever.dtype,
                "prefix_query": self.retriever.prefix_query,
                "prefix_passage": self.retriever.prefix_passage,
                "strategy": self.retriever.strategy,
                "normalize_whitespace": self.retriever.normalize_whitespace,
                "max_length": self.retriever.max_length,
            },
        }


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_top4_config(path: Path) -> Top4Config:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuração inválida: {path}")
    base_dir = path.parent
    dataset_raw = raw.get("dataset", {})
    retriever_raw = raw.get("retriever", {})
    dataset = DatasetSettings(
        response_path=_resolve_path(dataset_raw["response_path"], base_dir),
        source_path=_resolve_path(dataset_raw["source_path"], base_dir),
        revision=(str(dataset_raw["revision"]) if dataset_raw.get("revision") else None),
        task=str(dataset_raw.get("task", "QA")),
        splits=tuple(str(x) for x in dataset_raw.get("splits", ["train", "test"])),
        quality=str(dataset_raw.get("quality", "good")),
        granularity=str(dataset_raw.get("granularity", "claim")),
        min_words=int(dataset_raw.get("min_words", 3)),
        boundary_strategy=str(dataset_raw.get("boundary_strategy", "label_dependent_sentences")),
        segmentation_version=str(dataset_raw.get("segmentation_version", "sentence-spans-v1")),
        overlap_rule=str(dataset_raw.get("overlap_rule", "half_open_character_intersection-v1")),
        duplicate_policy=str(dataset_raw.get("duplicate_policy", "keep_all")),
    )
    retriever = RetrieverSettings(
        model_id=str(retriever_raw.get("model_id", "intfloat/multilingual-e5-base")),
        revision=(str(retriever_raw["revision"]) if retriever_raw.get("revision") else None),
        tokenizer_model_id=str(
            retriever_raw.get("tokenizer_model_id", retriever_raw.get("model_id", "intfloat/multilingual-e5-base"))
        ),
        tokenizer_revision=(
            str(retriever_raw["tokenizer_revision"])
            if retriever_raw.get("tokenizer_revision")
            else None
        ),
        encoder_backend=str(retriever_raw.get("encoder_backend", "transformers")),
        embedding_dim=int(retriever_raw.get("embedding_dim", 768)),
        chunk_size_tokens=int(retriever_raw.get("chunk_size_tokens", 256)),
        chunk_overlap_tokens=int(retriever_raw.get("chunk_overlap_tokens", 64)),
        batch_size=int(retriever_raw.get("batch_size", 64)),
        device=str(retriever_raw.get("device", "auto")),
        dtype=str(retriever_raw.get("dtype", "float32")),
        prefix_query=str(retriever_raw.get("prefix_query", "query: ")),
        prefix_passage=str(retriever_raw.get("prefix_passage", "passage: ")),
        strategy=str(retriever_raw.get("strategy", "topk_cosine")),
        normalize_whitespace=bool(retriever_raw.get("normalize_whitespace", True)),
        max_length=(int(retriever_raw["max_length"]) if retriever_raw.get("max_length") else None),
    )
    return Top4Config(
        run_name=str(raw.get("run_name", "ragtruth_qa_top4")),
        output_root=_resolve_path(raw.get("output_root", "results/ragtruth_qa_top4"), base_dir),
        cache_root=_resolve_path(raw.get("cache_root", "cache/ragtruth_qa_top4"), base_dir),
        dataset=dataset,
        retriever=retriever,
        top_k=int(raw.get("top_k", 4)),
        sample_size=int(raw.get("sample_size", 100)),
        audit_seed=int(raw.get("audit_seed", 42)),
        schema_version=str(raw.get("schema_version", "ragtruth-qa-top4-v1")),
        legacy_run_dir=(_resolve_path(raw["legacy_run_dir"], base_dir) if raw.get("legacy_run_dir") else None),
    )
