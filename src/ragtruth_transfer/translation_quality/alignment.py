from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..io_utils import sha256_file
from ..ragtruth_top4_embeddings import sha256_text
from .config import AlignmentConfig
from .storage import write_json_atomic, write_parquet_atomic

MANIFEST_SCHEMA = "ragtruth-translation-quality-alignment-manifest-v1"
OUTPUT_SCHEMA = "ragtruth-translation-quality-aligned-v1"


class AlignmentIntegrityError(RuntimeError):
    def __init__(self, message: str, counts: "AlignmentCounts", mismatches: list[dict[str, Any]]):
        super().__init__(message)
        self.counts = counts
        self.mismatches = mismatches


@dataclass
class AlignmentCounts:
    pt_rows: int = 0
    en_rows: int = 0
    en_extra_rows: int = 0
    missing_in_en: int = 0
    source_id_mismatches: int = 0
    label_mismatches: int = 0
    split_mismatches: int = 0
    response_id_mismatches: int = 0
    evidence_mask_mismatches: int = 0
    provenance_mismatches: int = 0
    en_chunk_sha256_mismatches: int = 0
    total_valid_chunks: int = 0
    en_chunk_sha256_verified: int = 0

    @property
    def gate_ok(self) -> bool:
        return (
            self.missing_in_en == 0
            and self.source_id_mismatches == 0
            and self.label_mismatches == 0
            and self.split_mismatches == 0
            and self.response_id_mismatches == 0
            and self.evidence_mask_mismatches == 0
            and self.provenance_mismatches == 0
            and self.en_chunk_sha256_mismatches == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pt_rows": self.pt_rows,
            "en_rows": self.en_rows,
            "en_extra_rows": self.en_extra_rows,
            "missing_in_en": self.missing_in_en,
            "source_id_mismatches": self.source_id_mismatches,
            "label_mismatches": self.label_mismatches,
            "split_mismatches": self.split_mismatches,
            "response_id_mismatches": self.response_id_mismatches,
            "evidence_mask_mismatches": self.evidence_mask_mismatches,
            "provenance_mismatches": self.provenance_mismatches,
            "en_chunk_sha256_mismatches": self.en_chunk_sha256_mismatches,
            "total_valid_chunks": self.total_valid_chunks,
            "en_chunk_sha256_verified": self.en_chunk_sha256_verified,
            "gate_ok": self.gate_ok,
        }


@dataclass
class AlignmentResult:
    config: AlignmentConfig
    counts: AlignmentCounts
    frame: pd.DataFrame
    mismatches: list[dict[str, Any]] = field(default_factory=list)


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} não contém as colunas obrigatórias: {missing}")


def _unique_or_raise(series: pd.Series, label: str) -> None:
    duplicated = series[series.duplicated()].unique()
    if len(duplicated):
        sample = list(duplicated[:5])
        raise ValueError(f"{label} possui example_id duplicado (exemplos: {sample}).")


def _coerce_mask(value: Any) -> list[bool]:
    if value is None:
        return []
    try:
        return [bool(x) for x in list(value)]
    except TypeError:
        return []


def _masks_equal(a: Any, b: Any) -> bool:
    left = _coerce_mask(a)
    right = _coerce_mask(b)
    return len(left) == len(right) and all(x == y for x, y in zip(left, right))


def _provenance_columns(config: AlignmentConfig) -> list[str]:
    columns: list[str] = []
    for chunk in config.chunk_columns:
        columns.extend(
            [
                f"{chunk}_source_index",
                f"{chunk}_window_index",
                f"{chunk}_token_start",
                f"{chunk}_token_end",
                f"{chunk}_sha256",
            ]
        )
    return columns


def _check_signature_guard(frame: pd.DataFrame, column: str, expected: str | None, label: str) -> None:
    if expected is None or column not in frame.columns:
        return
    observed = {str(x) for x in frame[column].dropna().unique()}
    if observed != {expected}:
        raise ValueError(
            f"{label}.{column} difere do esperado: {sorted(observed)} != {expected!r}"
        )


def align_translation_quality(
    config: AlignmentConfig, *, validate_only: bool = False
) -> AlignmentResult:
    en = pd.read_parquet(config.en_parquet)
    pt = pd.read_parquet(config.pt_parquet)

    shared_fields = [
        config.source_id_column,
        config.label_column,
        config.split_column,
        config.response_id_column,
        config.evidence_mask_column,
    ]
    provenance = _provenance_columns(config)
    en_required = [
        config.example_id_column,
        config.claim_column,
        *config.chunk_columns,
        *[c for c in shared_fields if c != config.evidence_mask_column],
    ]
    pt_required = [*en_required, config.evidence_mask_column]
    _require_columns(en, [*en_required, config.evidence_mask_column], "artefato EN")
    _require_columns(pt, pt_required, "artefato PT")

    if config.expected_rows is not None and len(pt) != config.expected_rows:
        raise ValueError(f"artefato PT possui {len(pt)} linhas, esperado {config.expected_rows}.")
    _check_signature_guard(pt, "chunking_signature", config.expected_chunking_signature, "artefato PT")
    _check_signature_guard(pt, "tokenizer_revision", config.expected_tokenizer_revision, "artefato PT")

    _unique_or_raise(pt[config.example_id_column].astype(str), "artefato PT")
    _unique_or_raise(en[config.example_id_column].astype(str), "artefato EN")

    counts = AlignmentCounts(pt_rows=len(pt), en_rows=len(en))
    mismatches: list[dict[str, Any]] = []

    pt_ids = set(pt[config.example_id_column].astype(str))
    en_ids = set(en[config.example_id_column].astype(str))
    missing = pt_ids - en_ids
    counts.missing_in_en = len(missing)
    counts.en_extra_rows = len(en_ids - pt_ids)
    for example_id in list(missing)[:1000]:
        mismatches.append({"example_id": example_id, "reason": "missing_in_en"})

    keep = [config.example_id_column, *shared_fields, config.claim_column, *config.chunk_columns, *provenance]
    if "task" in en.columns:
        keep.append("task")
    en_small = en[[c for c in keep if c in en.columns]].copy()
    pt_small = pt[[c for c in keep if c in pt.columns]].copy()
    en_small = en_small.rename(columns={c: f"{c}__en" for c in en_small.columns if c != config.example_id_column})
    pt_small = pt_small.rename(columns={c: f"{c}__pt" for c in pt_small.columns if c != config.example_id_column})
    merged = pt_small.merge(
        en_small, on=config.example_id_column, how="left", validate="one_to_one"
    )

    for field_name, counter in (
        (config.source_id_column, "source_id_mismatches"),
        (config.label_column, "label_mismatches"),
        (config.split_column, "split_mismatches"),
        (config.response_id_column, "response_id_mismatches"),
    ):
        left = merged[f"{field_name}__pt"].astype(str)
        right = merged[f"{field_name}__en"].astype(str)
        bad = merged[right.notna() & (left != right)]
        if len(bad):
            setattr(counts, counter, len(bad))
            for _, row in bad.head(1000).iterrows():
                mismatches.append(
                    {
                        "example_id": str(row[config.example_id_column]),
                        "reason": counter,
                        "pt": str(row[f"{field_name}__pt"]),
                        "en": str(row[f"{field_name}__en"]),
                    }
                )

    mask_mismatch_rows = merged[
        merged[f"{config.evidence_mask_column}__en"].notna()
        & merged.apply(
            lambda r: not _masks_equal(
                r[f"{config.evidence_mask_column}__pt"], r[f"{config.evidence_mask_column}__en"]
            ),
            axis=1,
        )
    ]
    counts.evidence_mask_mismatches = int(len(mask_mismatch_rows))
    for example_id in mask_mismatch_rows[config.example_id_column].head(1000):
        mismatches.append({"example_id": str(example_id), "reason": "evidence_mask"})

    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        example_id = str(row[config.example_id_column])
        record: dict[str, Any] = {
            "example_id": example_id,
            "source_id": row[f"{config.source_id_column}__pt"],
            "split": row[f"{config.split_column}__pt"],
            "label": row[f"{config.label_column}__pt"],
            "response_id": row[f"{config.response_id_column}__pt"],
            "evidence_mask": _coerce_mask(row[f"{config.evidence_mask_column}__pt"]),
            "claim_en": row[f"{config.claim_column}__en"],
            "claim_pt": row[f"{config.claim_column}__pt"],
        }
        if "task__pt" in merged.columns:
            record["task"] = row["task__pt"]

        en_mask = _coerce_mask(row[f"{config.evidence_mask_column}__en"])
        num_valid = 0
        for slot, chunk in enumerate(config.chunk_columns, start=1):
            valid = bool(en_mask[slot - 1]) if slot - 1 < len(en_mask) else False
            record[f"chunk_{slot}_valid"] = valid
            record[f"chunk_{slot}_en"] = row[f"{chunk}__en"]
            record[f"chunk_{slot}_pt"] = row[f"{chunk}__pt"]
            record[f"chunk_{slot}_sha256_verified"] = None
            if not valid:
                continue

            num_valid += 1
            counts.total_valid_chunks += 1
            chunk_en = "" if row[f"{chunk}__en"] is None else str(row[f"{chunk}__en"])
            expected_sha = str(row[f"{chunk}_sha256__pt"])
            actual_sha = sha256_text(chunk_en)
            if actual_sha == expected_sha:
                counts.en_chunk_sha256_verified += 1
                record[f"chunk_{slot}_sha256_verified"] = True
            else:
                counts.en_chunk_sha256_mismatches += 1
                record[f"chunk_{slot}_sha256_verified"] = False
                mismatches.append(
                    {
                        "example_id": example_id,
                        "slot": slot,
                        "reason": "en_chunk_sha256",
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                    }
                )

            for field_name in (
                "source_index",
                "window_index",
                "token_start",
                "token_end",
                "sha256",
            ):
                pt_value = str(row[f"{chunk}_{field_name}__pt"])
                en_value = str(row[f"{chunk}_{field_name}__en"])
                if pt_value != en_value:
                    counts.provenance_mismatches += 1
                    mismatches.append(
                        {
                            "example_id": example_id,
                            "slot": slot,
                            "reason": f"provenance_{field_name}",
                            "pt": pt_value,
                            "en": en_value,
                        }
                    )

        record["num_valid_chunks"] = num_valid
        records.append(record)

    out_frame = pd.DataFrame.from_records(records)
    result = AlignmentResult(config=config, counts=counts, frame=out_frame, mismatches=mismatches)

    if not counts.gate_ok:
        if not validate_only:
            config.run_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(config.run_dir / "mismatches.json", mismatches[:1000])
        raise AlignmentIntegrityError(
            "GATE DE INTEGRIDADE EN<->PT FALHOU: divergências de cobertura, ids, labels, "
            "proveniência ou sha256.",
            counts,
            mismatches,
        )

    if not validate_only:
        _write_outputs(result)
    return result


def _write_outputs(result: AlignmentResult) -> None:
    config = result.config
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    write_parquet_atomic(run_dir / "aligned.parquet", result.frame)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "signature": config.signature,
        "config": config.to_dict(),
        "inputs": {
            "en_parquet_sha256": sha256_file(config.en_parquet),
            "pt_parquet_sha256": sha256_file(config.pt_parquet),
        },
        "counts": result.counts.to_dict(),
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    write_json_atomic(run_dir / "resolved_config.json", config.to_dict())
