
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .io_utils import sha256_file

PARQUET_REQUIRED_COLUMNS = {
    "example_id", "source_id", "split", "claim", "label", "evidence_mask",
    "chunk_1", "chunk_2", "chunk_3", "chunk_4",
}
PARQUET_SCHEMA = "ragtruth-qa-training-view-deduplicated-v1"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [] if value is None or (isinstance(value, float) and np.isnan(value)) else [value]


def _binary_label(value: Any, example_id: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value) and float(value) in {0.0, 1.0}:
        return int(value)
    raise ValueError(f"label não binário no exemplo {example_id}: {value!r}")


def _manifest_for_dataset(path: Path, manifest_path: Path | None) -> tuple[dict[str, Any], Path | None]:
    resolved = manifest_path.resolve() if manifest_path else path.parent / "manifest.json"
    if not resolved.is_file():
        return {}, None
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Manifesto Parquet inválido: {resolved}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Manifesto Parquet deve ser um objeto JSON: {resolved}")
    return value, resolved


def validate_training_view_manifest(
    path: Path,
    *,
    manifest_path: Path | None = None,
    expected_signature: str | None = None,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parquet não encontrado: {path}")
    manifest, resolved_manifest = _manifest_for_dataset(path, manifest_path)
    if not manifest:
        raise ValueError("O Parquet precisa de manifest.json para validar lineage.")
    schema = str(manifest.get("schema_version", ""))
    if expected_schema and schema != expected_schema:
        raise ValueError(f"Schema Parquet incompatível: {schema!r}; esperado {expected_schema!r}")
    if expected_signature and str(manifest.get("signature", "")) != expected_signature:
        raise ValueError(
            f"Assinatura Parquet incompatível: {manifest.get('signature')!r}; esperado {expected_signature!r}"
        )
    expected_hash = manifest.get("artifacts", {}).get("dataset.parquet")
    actual_hash = sha256_file(path)
    if expected_hash and str(expected_hash) != actual_hash:
        raise ValueError("SHA-256 do dataset.parquet não coincide com o manifesto.")
    audit_path = path.parent / "training_view_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("labels_used_for_claim_boundaries") is not False:
            raise ValueError("A visão Parquet não confirma boundaries independentes dos labels.")
        for key in ("duplicate_groups_remaining", "conflicting_duplicate_groups_remaining", "input_mismatch_groups_included", "split_source_overlap_count"):
            if int(audit.get(key, 0)) != 0:
                raise ValueError(f"A auditoria Parquet viola o contrato: {key}={audit.get(key)}")
    return {
        "path": str(path),
        "manifest_path": str(resolved_manifest) if resolved_manifest else None,
        "manifest": manifest,
        "schema_version": schema,
        "signature": str(manifest.get("signature", "")),
        "dataset_sha256": actual_hash,
    }


def _validate_source_split_isolation(frame: pd.DataFrame) -> None:
    by_split = {str(split): set(group["source_id"].astype(str)) for split, group in frame.groupby("split")}
    splits = sorted(by_split)
    overlap = sorted(set.intersection(*(by_split[split] for split in splits))) if len(splits) > 1 else []
    if overlap:
        raise ValueError(f"source_id aparece em mais de um split: {overlap[:10]}")


def load_ragtruth_parquet(
    path: Path,
    *,
    manifest_path: Path | None = None,
    expected_signature: str | None = None,
    expected_schema: str | None = PARQUET_SCHEMA,
    claim_column: str = "claim",
    chunk_columns: tuple[str, ...] = ("chunk_1", "chunk_2", "chunk_3", "chunk_4"),
    evidence_mask_column: str = "evidence_mask",
    label_column: str = "label",
    group_column: str = "source_id",
    split_column: str = "split",
    source_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lineage = validate_training_view_manifest(
        path,
        manifest_path=manifest_path,
        expected_signature=expected_signature,
        expected_schema=expected_schema,
    )
    if len(chunk_columns) != 4:
        raise ValueError("chunk_columns deve conter exatamente quatro colunas")
    frame = pd.read_parquet(path)
    configured_columns = {
        "example_id": "example_id",
        "source_id": group_column,
        "split": split_column,
        "claim": claim_column,
        "label": label_column,
        "evidence_mask": evidence_mask_column,
        **{f"chunk_{index}": column for index, column in enumerate(chunk_columns, start=1)},
    }
    missing = sorted(set(configured_columns.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no Parquet: {missing}")
    frame = frame.rename(columns={value: key for key, value in configured_columns.items()})
    if source_ids is not None:
        requested_sources = {str(value) for value in source_ids}
        frame = frame.loc[frame["source_id"].astype(str).isin(requested_sources)].copy()
        if set(frame["source_id"].astype(str)) != requested_sources:
            raise ValueError("source_ids solicitado não corresponde ao Parquet")
    essential = ["example_id", "source_id", "split", "claim", "label", "evidence_mask", "chunk_1", "chunk_2", "chunk_3", "chunk_4"]
    if frame[essential].isna().any().any():
        bad = frame[essential].columns[frame[essential].isna().any()].tolist()
        raise ValueError(f"NaN em campos essenciais do Parquet: {bad}")
    frame = frame.copy()
    frame["example_id"] = frame["example_id"].astype(str)
    frame["source_id"] = frame["source_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["claim"] = frame["claim"].astype(str)
    if (frame["example_id"].str.strip() == "").any() or (frame["source_id"].str.strip() == "").any():
        raise ValueError("example_id e source_id não podem estar vazios no Parquet.")
    if (frame["claim"].str.strip() == "").any():
        raise ValueError("claim não pode estar vazia no Parquet.")
    if frame["example_id"].duplicated().any():
        raise ValueError("example_id duplicado no Parquet.")
    if not set(frame["split"]).issubset({"train", "test"}):
        raise ValueError(f"Split inválido no Parquet: {sorted(set(frame['split']) - {'train', 'test'})}")
    _validate_source_split_isolation(frame)
    rows: list[dict[str, Any]] = []
    for _, source in frame.iterrows():
        example_id = str(source["example_id"])
        evidence = [str(source[f"chunk_{index}"]) for index in range(1, 5)]
        raw_mask = _as_list(source["evidence_mask"])
        if any(not isinstance(value, (bool, np.bool_, int, np.integer)) or int(value) not in {0, 1} for value in raw_mask):
            raise ValueError(f"evidence_mask deve conter apenas booleanos/0/1: {example_id}")
        mask = [bool(value) for value in raw_mask]
        if len(mask) != 4:
            raise ValueError(f"evidence_mask deve ter quatro posições: {example_id}")
        if not any(mask):
            raise ValueError(f"Bag sem evidência válida: {example_id}")
        for index, valid in enumerate(mask):
            if not valid and evidence[index] != "":
                raise ValueError(f"Slot mascarado precisa estar vazio: {example_id}, chunk_{index + 1}")
            if valid and not evidence[index].strip():
                raise ValueError(f"Slot válido vazio: {example_id}, chunk_{index + 1}")
        rows.append({
            "example_id": example_id,
            "source_id": str(source["source_id"]),
            "split": str(source["split"]),
            "task_type": "QA",
            "claim": str(source["claim"]),
            "evidence": evidence,
            "evidence_mask": mask,
            "label": _binary_label(source["label"], example_id),
        })
    metadata = {
        "dataset_format": "parquet",
        "path": str(Path(path).resolve()),
        "manifest_path": lineage["manifest_path"],
        "schema_version": lineage["schema_version"],
        "signature": lineage["signature"],
        "dataset_sha256": lineage["dataset_sha256"],
        "rows": len(rows),
        "by_split": {str(key): int(value) for key, value in frame["split"].value_counts().sort_index().items()},
        "by_label": {str(key): int(value) for key, value in frame["label"].astype(int).value_counts().sort_index().items()},
        "sources": int(frame["source_id"].nunique()),
        "feature_columns": ["claim", "chunk_1", "chunk_2", "chunk_3", "chunk_4", "evidence_mask", "label", "source_id", "example_id", "split"],
        "excluded_columns": ["*_score", "span_*", "duplicate_*", "model", "response_id", "retrieval_signature", "chunking_signature"],
    }
    return rows, metadata


@dataclass(frozen=True)
class ParquetSplit:
    train_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    assignments: pd.DataFrame
    metadata: dict[str, Any]


def _split_signature(dataset_sha256: str, view_signature: str, validation_fraction: float, seed: int, version: str) -> str:
    payload = json.dumps({"dataset_sha256": dataset_sha256, "view_signature": view_signature, "group_key": "source_id", "strategy": "stratified_group_shuffle_search", "validation_fraction": validation_fraction, "seed": seed, "version": version}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    sources = {str(row["source_id"]) for row in rows}
    return {"sources": len(sources), "examples": len(rows), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()), "prevalence": float(labels.mean()) if len(labels) else 0.0, "largest_source_examples": max((sum(str(row["source_id"]) == source for row in rows) for source in sources), default=0)}


def _limited_sources(source_ids: list[str], group_positive: dict[str, bool], limit: int | None) -> set[str]:
    if limit is None or limit >= len(source_ids):
        return set(source_ids)
    limit = max(1, int(limit))
    positives = sorted(source for source in source_ids if group_positive[source])
    negatives = sorted(source for source in source_ids if not group_positive[source])
    chosen: list[str] = []
    if positives:
        chosen.append(positives[0])
    if negatives and len(chosen) < limit:
        chosen.append(negatives[0])
    for source in sorted(set(source_ids) - set(chosen)):
        if len(chosen) >= limit:
            break
        chosen.append(source)
    return set(chosen)


def split_ragtruth_parquet(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    validation_fraction: float = 0.15,
    split_seed: int = 42,
    max_train_sources: int | None = None,
    max_validation_sources: int | None = None,
    max_test_sources: int | None = None,
    algorithm_version: str = "group-split-v1",
) -> ParquetSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    train_rows_all = [row for row in rows if row["split"] == "train"]
    test_rows_all = [row for row in rows if row["split"] == "test"]
    test_source_ids = sorted({str(row["source_id"]) for row in test_rows_all})
    if max_test_sources is None:
        test_rows = test_rows_all
    elif max_test_sources <= 0:
        test_rows = []
    else:
        test_rows = [row for row in test_rows_all if str(row["source_id"]) in set(test_source_ids[:max_test_sources])]
    group_rows: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows_all:
        group_rows.setdefault(str(row["source_id"]), []).append(row)
    source_ids = sorted(group_rows)
    if len(source_ids) < 4:
        raise ValueError("São necessárias pelo menos quatro fontes no split train para validation agrupada.")
    group_positive = {source: any(bool(row["label"]) for row in group_rows[source]) for source in source_ids}
    strata = np.asarray([int(group_positive[source]) for source in source_ids])
    if len(set(strata)) < 2 or min(int((strata == value).sum()) for value in (0, 1)) < 2:
        raise ValueError("A divisão agrupada exige ao menos duas fontes com e sem positivos.")
    validation_size = max(2, int(math.ceil(validation_fraction * len(source_ids))))
    if validation_size >= len(source_ids):
        validation_size = len(source_ids) - 2
    splitter = StratifiedShuffleSplit(n_splits=128, test_size=validation_size, random_state=split_seed)
    total_train_labels = np.asarray([int(row["label"]) for row in train_rows_all])
    target_prevalence = float(total_train_labels.mean())
    best: tuple[float, np.ndarray] | None = None
    for train_index, validation_index in splitter.split(source_ids, strata):
        validation_sources = {source_ids[index] for index in validation_index}
        validation_rows_candidate = [row for row in train_rows_all if str(row["source_id"]) in validation_sources]
        validation_prevalence = float(np.mean([int(row["label"]) for row in validation_rows_candidate]))
        source_fraction = len(validation_sources) / len(source_ids)
        example_fraction = len(validation_rows_candidate) / len(train_rows_all)
        objective = abs(source_fraction - validation_fraction) + abs(example_fraction - validation_fraction) + 2.0 * abs(validation_prevalence - target_prevalence)
        candidate = (objective, np.asarray(sorted(validation_index)))
        if best is None or (candidate[0], tuple(candidate[1])) < (best[0], tuple(best[1])):
            best = candidate
    if best is None:
        raise RuntimeError("Não foi possível criar split agrupado determinístico.")
    validation_sources = {source_ids[index] for index in best[1]}
    train_sources = set(source_ids) - validation_sources
    train_sources = _limited_sources(sorted(train_sources), group_positive, max_train_sources)
    validation_sources = _limited_sources(sorted(validation_sources), group_positive, max_validation_sources)
    train_rows = [row for row in train_rows_all if str(row["source_id"]) in train_sources]
    validation_rows = [row for row in train_rows_all if str(row["source_id"]) in validation_sources]
    if not train_rows or not validation_rows:
        raise ValueError("Limitação por fontes produziu treino ou validation vazio.")
    if len({row["source_id"] for row in train_rows} & {row["source_id"] for row in validation_rows}):
        raise RuntimeError("Falha de isolamento: fonte presente em train e validation.")
    if len({row["source_id"] for row in test_rows} & ({row["source_id"] for row in train_rows} | {row["source_id"] for row in validation_rows})):
        raise RuntimeError("Falha de isolamento: fonte do test presente em train/validation.")
    assignments: list[dict[str, Any]] = []
    for source in source_ids:
        partition = "validation" if source in validation_sources else ("train" if source in train_sources else "excluded_smoke_limit")
        source_group = group_rows[source]
        labels = [int(row["label"]) for row in source_group]
        assignments.append({"source_id": source, "partition": partition, "examples": len(source_group), "positives": int(sum(labels)), "negatives": int(len(labels) - sum(labels)), "split_seed": int(split_seed)})
    split_sig = _split_signature(str(metadata["dataset_sha256"]), str(metadata.get("signature", "")), validation_fraction, split_seed, algorithm_version)
    assignments_frame = pd.DataFrame(assignments)
    assignments_frame["split_signature"] = split_sig
    assignments_frame["algorithm_version"] = algorithm_version
    assignment_path = None
    split_metadata = {"strategy": "stratified_group_shuffle_search", "algorithm_version": algorithm_version, "group_key": "source_id", "validation_fraction": validation_fraction, "split_seed": split_seed, "signature": split_sig, "train": _stats(train_rows), "validation": _stats(validation_rows), "test": _stats(test_rows), "source_overlap_train_validation": len(set(row["source_id"] for row in train_rows) & set(row["source_id"] for row in validation_rows)), "source_overlap_train_test": len(set(row["source_id"] for row in train_rows) & set(row["source_id"] for row in test_rows)), "source_overlap_validation_test": len(set(row["source_id"] for row in validation_rows) & set(row["source_id"] for row in test_rows)), "validation_prevalence_delta_from_train": abs(_stats(train_rows)["prevalence"] - _stats(validation_rows)["prevalence"]), "validation_source_fraction_delta": abs(_stats(validation_rows)["sources"] / len(source_ids) - validation_fraction), "assignment_path": assignment_path}
    return ParquetSplit(train_rows, validation_rows, test_rows, assignments_frame, split_metadata)
