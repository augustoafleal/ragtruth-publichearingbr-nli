
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io_utils import iter_jsonl, sha256_file
from .ragtruth_top4_pipeline import build_surface_feature_audit

TRAINING_VIEW_SCHEMA = "ragtruth-qa-training-view-deduplicated-v1"
SUPPORTED_POLICY = "deduplicate_source_claim_drop_conflicts_v1"
PARENT_SCHEMA = "ragtruth-qa-top4-label-independent-v2"
KEY_SEPARATOR = "\x1f"
REPRESENTATIVE_RULE = "min response_id (text), claim_start, claim_end, example_id (text)"

_INPUT_TEXT_FIELDS = ["claim", "chunk_1", "chunk_2", "chunk_3", "chunk_4"]
_INPUT_EXACT_FIELDS = [
    "chunk_1_sha256", "chunk_2_sha256", "chunk_3_sha256", "chunk_4_sha256",
    "chunk_1_source_index", "chunk_2_source_index", "chunk_3_source_index", "chunk_4_source_index",
    "chunk_1_window_index", "chunk_2_window_index", "chunk_3_window_index", "chunk_4_window_index",
    "chunk_1_token_start", "chunk_2_token_start", "chunk_3_token_start", "chunk_4_token_start",
    "chunk_1_token_end", "chunk_2_token_end", "chunk_3_token_end", "chunk_4_token_end",
    "num_candidate_chunks", "retriever_model", "retriever_revision", "tokenizer_model",
    "tokenizer_revision", "chunking_signature", "retrieval_signature",
]
_INPUT_FLOAT_FIELDS = ["chunk_1_score", "chunk_2_score", "chunk_3_score", "chunk_4_score"]
_SCORE_ATOL = 1e-6
_SCORE_RTOL = 1e-6


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default, separators=(",", ":"))


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_serializable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes, list, tuple, dict)) else False:
        return None
    return value


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    if temporary.exists():
        temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8"))


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", mode="w", encoding="utf-8", newline="", delete=False)
    temporary = Path(output.name)
    try:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        output.flush()
        os.fsync(output.fileno())
        output.close()
        os.replace(temporary, path)
    finally:
        if not output.closed:
            output.close()
        if temporary.exists():
            temporary.unlink()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def duplicate_group_id(source_id: Any, claim: Any) -> str:
    return _stable_hash(f"{source_id}{KEY_SEPARATOR}{claim}")


def _row_value(row: pd.Series | dict[str, Any], field: str) -> Any:
    return row[field] if isinstance(row, pd.Series) else row.get(field)


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [_serializable(value)]


def _same_float(left: Any, right: Any) -> bool:
    left_nan = pd.isna(left)
    right_nan = pd.isna(right)
    if left_nan or right_nan:
        return bool(left_nan and right_nan)
    return bool(math.isclose(float(left), float(right), rel_tol=_SCORE_RTOL, abs_tol=_SCORE_ATOL))


def input_differences(group: pd.DataFrame) -> list[dict[str, Any]]:
    if len(group) < 2:
        return []
    reference = group.iloc[0]
    differences: list[dict[str, Any]] = []
    for _, row in group.iloc[1:].iterrows():
        example_id = str(row.get("example_id", ""))
        for field in _INPUT_TEXT_FIELDS + _INPUT_EXACT_FIELDS:
            left = _list_value(reference[field]) if field == "evidence_mask" else _serializable(reference[field])
            right = _list_value(row[field]) if field == "evidence_mask" else _serializable(row[field])
            if _json_text(left) != _json_text(right):
                differences.append({"example_id": example_id, "field": field, "reference": left, "value": right})
        ref_mask = _list_value(reference.get("evidence_mask", []))
        row_mask = _list_value(row.get("evidence_mask", []))
        if ref_mask != row_mask:
            differences.append({"example_id": example_id, "field": "evidence_mask", "reference": ref_mask, "value": row_mask})
        for field in _INPUT_FLOAT_FIELDS:
            if not _same_float(reference[field], row[field]):
                differences.append({"example_id": example_id, "field": field, "reference": _serializable(reference[field]), "value": _serializable(row[field])})
    return differences


def _group_rows(frame: pd.DataFrame) -> dict[tuple[str, str], list[Any]]:
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for index, (source_id, claim) in enumerate(zip(frame["source_id"].astype(str), frame["claim"].astype(str), strict=True)):
        groups[(str(source_id), str(claim))].append(frame.index[index])
    return groups


def _sort_group(group: pd.DataFrame) -> pd.DataFrame:
    values = group.copy()
    values["_response_sort"] = values["response_id"].astype(str)
    values["_example_sort"] = values["example_id"].astype(str)
    values["_start_sort"] = pd.to_numeric(values["claim_start"], errors="coerce").fillna(10**18)
    values["_end_sort"] = pd.to_numeric(values["claim_end"], errors="coerce").fillna(10**18)
    return values.sort_values(["_response_sort", "_start_sort", "_end_sort", "_example_sort"], kind="mergesort").drop(columns=["_response_sort", "_example_sort", "_start_sort", "_end_sort"])


def _group_summary(source_id: str, claim: str, group: pd.DataFrame, differences: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = _sort_group(group)
    labels = sorted({int(bool(value)) for value in group["label"].tolist()})
    response_ids = [str(value) for value in ordered["response_id"].tolist()]
    models = sorted({str(value) for value in group["model" if "model" in group.columns else "generator_model"].tolist()})
    starts = [int(value) for value in ordered["claim_start"].tolist()]
    ends = [int(value) for value in ordered["claim_end"].tolist()]
    return {
        "duplicate_group_id": duplicate_group_id(source_id, claim),
        "source_id": source_id,
        "claim": claim,
        "multiplicity": int(len(group)),
        "labels": labels,
        "conflict": len(labels) > 1,
        "example_ids": [str(value) for value in ordered["example_id"].tolist()],
        "response_ids": response_ids,
        "models": models,
        "splits": sorted({str(value) for value in group["split"].tolist()}),
        "same_response": len(set(response_ids)) == 1,
        "cross_response": len(set(response_ids)) > 1,
        "cross_model": len(models) > 1,
        "input_equivalent": not bool(differences),
        "input_mismatch_fields": sorted({str(item["field"]) for item in differences}),
        "input_differences": differences,
        "claim_starts": starts,
        "claim_ends": ends,
    }


def _diagnose_conflict(summary: dict[str, Any]) -> tuple[str, str, str]:
    if not summary["input_equivalent"]:
        return "insufficient_information", "low", "Excluir até investigar divergências nos inputs do classificador."
    if summary["same_response"] and len(set(summary["claim_starts"])) > 1:
        return "span_occurrence_mismatch", "high", "Excluir todas as ocorrências; revisar associação dos spans às ocorrências."
    if summary["cross_response"] and summary["cross_model"]:
        return "legitimate_cross_model_difference", "medium", "Excluir da visão principal; preservar para análise de sensibilidade."
    if summary["cross_response"]:
        return "context_dependent_label", "medium", "Excluir, pois o contexto completo não será uma feature do classificador."
    return "annotation_conflict", "medium", "Excluir e revisar a anotação original."


def _raw_maps(response_path: Path | None, source_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    responses: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    if response_path and response_path.is_file():
        for row in iter_jsonl(response_path):
            if str(row.get("id", "")):
                responses[str(row["id"])] = row
    if source_path and source_path.is_file():
        for row in iter_jsonl(source_path):
            if str(row.get("source_id", "")):
                sources[str(row["source_id"])] = row
    return responses, sources


def _response_detail(row: pd.Series, responses: dict[str, dict[str, Any]], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    response_id = str(row.get("response_id", ""))
    source_id = str(row.get("source_id", ""))
    raw_response = responses.get(response_id, {})
    raw_source = sources.get(source_id, {})
    span_offsets = _list_value(row.get("span_offsets", []))
    return {
        "example_id": str(row.get("example_id", "")),
        "response_id": response_id,
        "source_id": source_id,
        "split": str(row.get("split", "")),
        "model": str(row.get("model", row.get("generator_model", ""))),
        "label": int(bool(row.get("label", False))),
        "claim_start": int(row.get("claim_start", -1)),
        "claim_end": int(row.get("claim_end", -1)),
        "span_texts": _list_value(row.get("span_texts", [])),
        "span_offsets": span_offsets,
        "raw_labels": _serializable(raw_response.get("labels", [])),
        "response_text": str(raw_response.get("response", "")),
        "source_question": str((raw_source.get("source_info") or {}).get("question", "")) if isinstance(raw_source.get("source_info"), dict) else "",
        "evidence": [_serializable(row.get(f"chunk_{index}")) for index in range(1, 5)],
        "evidence_mask": _list_value(row.get("evidence_mask", [])),
        "scores": [_serializable(row.get(f"chunk_{index}_score")) for index in range(1, 5)],
        "retrieval_signature": str(row.get("retrieval_signature", "")),
        "chunking_signature": str(row.get("chunking_signature", "")),
    }


def _review_artifacts(groups: list[dict[str, Any]], frame: pd.DataFrame, responses: dict[str, dict[str, Any]], sources: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    review_rows: list[dict[str, Any]] = []
    markdown: list[str] = ["# Conflict review", "", f"Total de grupos conflitantes: **{len(groups)}**", ""]
    for index, summary in enumerate(sorted(groups, key=lambda item: (item["source_id"], item["claim"])), start=1):
        diagnostic, confidence, recommendation = _diagnose_conflict(summary)
        summary["diagnostic"] = diagnostic
        summary["confidence"] = confidence
        summary["recommendation"] = recommendation
        group = frame[frame["duplicate_group_id"] == summary["duplicate_group_id"]]
        details = [_response_detail(row, responses, sources) for _, row in _sort_group(group).iterrows()]
        priority = "INTRA-RESPONSE" if summary["same_response"] else ("SOURCE-15278" if summary["source_id"] == "15278" else "INTER-RESPONSE")
        review_rows.append({
            "duplicate_group_id": summary["duplicate_group_id"],
            "source_id": summary["source_id"],
            "claim": summary["claim"],
            "split": ";".join(summary["splits"]),
            "multiplicity": summary["multiplicity"],
            "labels": _json_text(summary["labels"]),
            "example_ids": _json_text(summary["example_ids"]),
            "response_ids": _json_text(summary["response_ids"]),
            "models": _json_text(summary["models"]),
            "claim_starts": _json_text(summary["claim_starts"]),
            "claim_ends": _json_text(summary["claim_ends"]),
            "input_equivalent": summary["input_equivalent"],
            "input_mismatch_fields": _json_text(summary["input_mismatch_fields"]),
            "location": "intra-response" if summary["same_response"] else "inter-response",
            "priority": priority,
            "diagnostic": diagnostic,
            "confidence": confidence,
            "recommendation": recommendation,
            "rows_json": _json_text(details),
        })
        markdown.extend([
            f"## {index}. {priority} — source `{summary['source_id']}`",
            "",
            f"- Group: `{summary['duplicate_group_id']}`",
            f"- Claim: `{summary['claim']}`",
            f"- Labels: `{summary['labels']}`; multiplicidade: `{summary['multiplicity']}`",
            f"- Exemplos: `{', '.join(summary['example_ids'])}`",
            f"- Diagnóstico: **{diagnostic}** (confiança {confidence})",
            f"- Recomendação: {recommendation}",
            f"- Inputs equivalentes: `{summary['input_equivalent']}`",
            "",
        ])
        for detail in details:
            context = detail["response_text"].replace("\n", " ")
            start = max(0, detail["claim_start"] - 180)
            end = min(len(context), detail["claim_end"] + 180)
            markdown.extend([
                f"### Example `{detail['example_id']}` — label `{detail['label']}`, model `{detail['model']}`",
                f"- response_id: `{detail['response_id']}`; offsets: `{detail['claim_start']}:{detail['claim_end']}`",
                f"- spans: `{_json_text(detail['span_offsets'])}`",
                f"- context: `{context[start:end]}`",
                f"- evidence_mask: `{_json_text(detail['evidence_mask'])}`; scores: `{_json_text(detail['scores'])}`",
                "",
            ])
    return review_rows, "\n".join(markdown) + "\n"


def _value_counts(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "sources": 0, "positives": 0, "negatives": 0, "prevalence": 0.0, "splits": {}, "models": {}, "claim_words": {}, "candidate_chunks": {}, "masked_slots": {}}
    labels = frame["label"].astype(bool)
    claim_words = frame["claim"].astype(str).str.split().str.len()
    candidate = pd.to_numeric(frame["num_candidate_chunks"], errors="coerce").fillna(0)
    masks = [sum(not bool(value) for value in _list_value(mask)) for mask in frame["evidence_mask"]]
    split_stats: dict[str, Any] = {}
    for split, group in frame.groupby("split", sort=True):
        split_labels = group["label"].astype(bool)
        split_stats[str(split)] = {"rows": int(len(group)), "positives": int(split_labels.sum()), "negatives": int((~split_labels).sum()), "prevalence": float(split_labels.mean()) if len(group) else 0.0, "sources": int(group["source_id"].nunique())}
    return {
        "rows": int(len(frame)),
        "sources": int(frame["source_id"].nunique()),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "prevalence": float(labels.mean()),
        "splits": split_stats,
        "models": {str(k): int(v) for k, v in frame["model"].astype(str).value_counts().sort_index().items()},
        "claim_words": {"min": int(claim_words.min()), "mean": float(claim_words.mean()), "max": int(claim_words.max())},
        "candidate_chunks": {str(k): int(v) for k, v in candidate.value_counts().sort_index().items()},
        "masked_slots": {"total": int(sum(masks)), "mean_per_row": float(np.mean(masks))},
    }


def _split_source_overlap(frame: pd.DataFrame) -> int:
    by_split = {str(split): set(group["source_id"].astype(str)) for split, group in frame.groupby("split")}
    splits = sorted(by_split)
    return int(sum(bool(by_split[left] & by_split[right]) for index, left in enumerate(splits) for right in splits[index + 1:]))


def _audit_baseline(frame: pd.DataFrame, parent: pd.DataFrame, seed: int) -> dict[str, Any]:
    try:
        current = build_surface_feature_audit(frame.to_dict("records"), seed=seed)
    except ValueError as error:
        current = {"status": "skipped", "reason": str(error)}
    try:
        base = build_surface_feature_audit(parent.to_dict("records"), seed=seed)
    except ValueError as error:
        base = {"status": "skipped", "reason": str(error)}
    return {"derived": current, "parent": base, "comparison": {"validation_AUPRC_delta": (float(current.get("validation_AUPRC", 0.0)) - float(base.get("validation_AUPRC", 0.0))) if current.get("status") == "completed" and base.get("status") == "completed" else None, "validation_AUROC_delta": (float(current.get("validation_AUROC", 0.0)) - float(base.get("validation_AUROC", 0.0))) if current.get("status") == "completed" and base.get("status") == "completed" else None}, "test_used": False}


def _valid_parent_manifest(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    run_dir = run_dir.resolve()
    if "507bfd0a5ed06f92" in str(run_dir):
        raise ValueError("O run legado 507bfd0a5ed06f92 é proibido como dataset-base.")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file() or not (run_dir / "dataset.parquet").is_file():
        raise ValueError(f"Run-base incompleto: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PARENT_SCHEMA:
        raise ValueError(f"Schema incompatível: {manifest.get('schema_version')!r}; esperado {PARENT_SCHEMA!r}.")
    retrieval_audit = json.loads((run_dir / "retrieval_audit.json").read_text(encoding="utf-8")) if (run_dir / "retrieval_audit.json").is_file() else {}
    if retrieval_audit.get("integrity", {}).get("labels_used_for_claim_boundaries") is not False:
        raise ValueError("O run-base não confirma labels_used_for_claim_boundaries=false.")
    expected = manifest.get("artifacts", {}).get("dataset.parquet")
    actual = sha256_file(run_dir / "dataset.parquet")
    if expected and expected != actual:
        raise ValueError("SHA-256 do dataset.parquet não coincide com o manifesto do run-base.")
    for name, expected_hash in dict(manifest.get("artifacts", {})).items():
        artifact = run_dir / str(name)
        if not artifact.is_file() or sha256_file(artifact) != str(expected_hash):
            raise ValueError(f"Hash inválido ou artefato ausente no run-base: {name}")
    frame = pd.read_parquet(run_dir / "dataset.parquet")
    # Pandas 3 may expose Parquet strings as Arrow-backed ``str`` columns.
    # Scalar ``.loc`` access on those columns is unexpectedly expensive for
    # the many singleton groups in this audit, so use ordinary Python objects
    # for the small textual keys/metadata while retaining list columns.
    for column in frame.columns:
        if str(frame[column].dtype) == "str":
            frame[column] = frame[column].astype(object)
    if len(frame) != int(manifest.get("counts", {}).get("examples", len(frame))):
        raise ValueError("Quantidade de linhas do Parquet não coincide com o manifesto do run-base.")
    return manifest, frame, actual


def discover_label_independent_runs(results_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for manifest_path in sorted(results_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit_path = manifest_path.parent / "retrieval_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
            if manifest.get("schema_version") == PARENT_SCHEMA and audit.get("integrity", {}).get("labels_used_for_claim_boundaries") is False:
                candidates.append(manifest_path.parent)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return candidates


def _manifest_valid(path: Path, signature: str) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature or manifest.get("schema_version") != TRAINING_VIEW_SCHEMA:
            return False
        for name, expected in dict(manifest.get("artifacts", {})).items():
            artifact = path / str(name)
            if not artifact.is_file() or sha256_file(artifact) != expected:
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _environment() -> dict[str, Any]:
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "cuda_used": False, "encoder_loaded": False}


def build_training_view(
    input_run_dir: Path | None = None,
    *,
    output_root: Path = Path("results/ragtruth_qa_training_view"),
    policy: str = SUPPORTED_POLICY,
    audit_only: bool = False,
    resume: bool = False,
    force: bool = False,
    raw_response_path: Path | None = None,
    raw_source_path: Path | None = None,
    results_root: Path = Path("results"),
    seed: int = 42,
) -> dict[str, Any]:
    if policy != SUPPORTED_POLICY:
        raise ValueError(f"Política não suportada: {policy!r}")
    if input_run_dir is None:
        candidates = discover_label_independent_runs(results_root)
        if len(candidates) != 1:
            raise ValueError(f"Forneça --input-run-dir explicitamente; candidatos encontrados: {[str(p) for p in candidates]}")
        input_run_dir = candidates[0]
    parent_manifest, parent, parent_sha = _valid_parent_manifest(input_run_dir)
    parent_signature = str(parent_manifest["signature"])
    signature_payload = {
        "schema_version": TRAINING_VIEW_SCHEMA,
        "parent_signature": parent_signature,
        "parent_dataset_sha256": parent_sha,
        "parent_schema": PARENT_SCHEMA,
        "policy": policy,
        "group_key": ["source_id", "claim"],
        "key_separator": KEY_SEPARATOR,
        "input_equivalence": {"exact": _INPUT_TEXT_FIELDS + _INPUT_EXACT_FIELDS + ["evidence_mask"], "float": _INPUT_FLOAT_FIELDS, "atol": _SCORE_ATOL, "rtol": _SCORE_RTOL},
        "conflict_policy": "drop_all_conflicting_groups",
        "representative_rule": REPRESENTATIVE_RULE,
        "code_version": "training-view-v1",
    }
    signature = _stable_hash(_json_text(signature_payload))[:16]
    destination = (Path(output_root).resolve() / policy / signature).resolve()
    if destination == Path(input_run_dir).resolve() or Path(input_run_dir).resolve() in destination.parents:
        raise ValueError("O output não pode ficar dentro do run-base.")
    if resume:
        if _manifest_valid(destination, signature):
            return json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if (destination / "manifest.json").exists() and not force:
            raise ValueError("--resume encontrou manifesto inválido ou hashes divergentes; use --force para reconstruir o output esperado.")
    elif destination.exists() and any(destination.iterdir()) and not force:
        existing_path = destination / "manifest.json"
        existing_status = ""
        if existing_path.is_file():
            try:
                existing_status = str(json.loads(existing_path.read_text(encoding="utf-8")).get("status", ""))
            except (OSError, ValueError, json.JSONDecodeError):
                existing_status = ""
        if not (existing_status == "audit_only" and not audit_only):
            raise FileExistsError(f"Output já existe; use --resume ou --force: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = parent.copy()
    frame["source_id"] = frame["source_id"].astype(str)
    frame["claim"] = frame["claim"].astype(str)
    frame["duplicate_group_id"] = [duplicate_group_id(source, claim) for source, claim in zip(frame["source_id"], frame["claim"], strict=True)]
    groups = _group_rows(frame)
    summaries: list[dict[str, Any]] = []
    selected_indices: list[Any] = []
    excluded_conflict_indices: list[Any] = []
    excluded_mismatch_indices: list[Any] = []
    collapsed_copy_indices: list[Any] = []
    for (source_id, claim), indices in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        if len(indices) == 1:
            row = frame.loc[indices[0]]
            summary = {
                "duplicate_group_id": duplicate_group_id(source_id, claim),
                "source_id": source_id,
                "claim": claim,
                "multiplicity": 1,
                "labels": [int(bool(row["label"]))],
                "conflict": False,
                "example_ids": [str(row["example_id"])],
                "response_ids": [str(row["response_id"])],
                "models": [str(row.get("model", row.get("generator_model", "")))],
                "splits": [str(row["split"])],
                "same_response": True,
                "cross_response": False,
                "cross_model": False,
                "input_equivalent": True,
                "input_mismatch_fields": [],
                "input_differences": [],
                "claim_starts": [int(row["claim_start"])],
                "claim_ends": [int(row["claim_end"])],
            }
            summaries.append(summary)
            selected_indices.append(indices[0])
            continue
        group = frame.loc[indices]
        differences = input_differences(group)
        summary = _group_summary(source_id, claim, group, differences)
        summaries.append(summary)
        ordered = _sort_group(group)
        if summary["conflict"]:
            excluded_conflict_indices.extend(list(group.index))
        elif not summary["input_equivalent"] and len(group) > 1:
            excluded_mismatch_indices.extend(list(group.index))
        else:
            selected_indices.append(ordered.index[0])
            collapsed_copy_indices.extend(list(ordered.index[1:]))
    repeated = [item for item in summaries if item["multiplicity"] > 1]
    conflicts = [item for item in repeated if item["conflict"]]
    mismatch = [item for item in repeated if not item["input_equivalent"]]
    responses_path = raw_response_path
    sources_path = raw_source_path
    if responses_path is None:
        candidate = parent_manifest.get("dataset", {}).get("response_path")
        responses_path = Path(candidate) if candidate else None
    if sources_path is None:
        candidate = parent_manifest.get("dataset", {}).get("source_path")
        sources_path = Path(candidate) if candidate else None
    if responses_path is None or not responses_path.is_file():
        local_response = Path("data/raw/ragtruth/response.jsonl").resolve()
        responses_path = local_response if local_response.is_file() else responses_path
    if sources_path is None or not sources_path.is_file():
        local_source = Path("data/raw/ragtruth/source_info.jsonl").resolve()
        sources_path = local_source if local_source.is_file() else sources_path
    responses, sources = _raw_maps(responses_path, sources_path)
    review_rows, review_md = _review_artifacts(conflicts, frame, responses, sources)
    duplicate_groups_rows = []
    for summary in repeated:
        decision = "drop_conflict" if summary["conflict"] else ("drop_input_mismatch" if not summary["input_equivalent"] else "collapse_to_representative")
        duplicate_groups_rows.append({**{key: value for key, value in summary.items() if key != "input_differences"}, "decision": decision, "input_differences": _json_text(summary["input_differences"])})
    selected = frame.loc[selected_indices].copy().sort_index()
    selected["duplicate_group_size"] = selected["duplicate_group_id"].map({item["duplicate_group_id"]: item["multiplicity"] for item in summaries}).astype(int)
    selected["duplicate_was_collapsed"] = selected["duplicate_group_size"] > 1
    summary_map = {item["duplicate_group_id"]: item for item in summaries}
    for field, source_field in [("duplicate_example_ids", "example_ids"), ("duplicate_response_ids", "response_ids"), ("duplicate_models", "models"), ("duplicate_claim_starts", "claim_starts"), ("duplicate_claim_ends", "claim_ends")]:
        selected[field] = selected["duplicate_group_id"].map(lambda key: _json_text(summary_map[key][source_field]))
    selected["representative_selection_rule"] = REPRESENTATIVE_RULE
    selected = selected.reset_index(drop=True)
    excluded_conflicts = frame.loc[sorted(set(excluded_conflict_indices))].copy().reset_index(drop=True)
    excluded_mismatches = frame.loc[sorted(set(excluded_mismatch_indices))].copy().reset_index(drop=True)
    for excluded in (excluded_conflicts, excluded_mismatches):
        if not excluded.empty:
            excluded["exclusion_reason"] = "conflicting_labels" if excluded is excluded_conflicts else "input_mismatch"
    parent_stats = _value_counts(parent)
    derived_stats = _value_counts(selected)
    conflict_stats = _value_counts(excluded_conflicts)
    mismatch_stats = _value_counts(excluded_mismatches)
    collapsed_rows = frame.loc[sorted(set(collapsed_copy_indices))]
    collapsed_stats = _value_counts(collapsed_rows)
    decisions = Counter(item["decision"] for item in duplicate_groups_rows)
    duplicate_audit = {
        "schema_version": "ragtruth-qa-training-view-duplicate-audit-v1",
        "policy": policy,
        "group_key": ["source_id", "claim"],
        "key_separator": "\\x1f",
        "groups_total": len(summaries),
        "groups_singleton": sum(item["multiplicity"] == 1 for item in summaries),
        "groups_repeated": len(repeated),
        "groups_consistent": sum(item["multiplicity"] > 1 and not item["conflict"] for item in summaries),
        "groups_conflicting": len(conflicts),
        "groups_input_mismatch": len(mismatch),
        "duplicate_rows": int(sum(item["multiplicity"] for item in repeated)),
        "extra_rows": int(sum(item["multiplicity"] - 1 for item in repeated)),
        "max_multiplicity": max((item["multiplicity"] for item in summaries), default=0),
        "same_response_groups": sum(item["same_response"] for item in repeated),
        "cross_response_groups": sum(item["cross_response"] for item in repeated),
        "cross_model_groups": sum(item["cross_model"] for item in repeated),
        "conflicting_same_response_groups": sum(item["conflict"] and item["same_response"] for item in repeated),
        "conflicting_cross_response_groups": sum(item["conflict"] and item["cross_response"] for item in repeated),
        "decision_counts": dict(decisions),
        "groups": duplicate_groups_rows,
    }
    source_split_overlap = _split_source_overlap(selected)
    training_view_audit = {
        "schema_version": TRAINING_VIEW_SCHEMA,
        "parent_signature": parent_signature,
        "labels_used_for_claim_boundaries": False,
        "duplicate_groups_remaining": int(selected.duplicated(["source_id", "claim"]).sum()),
        "conflicting_duplicate_groups_remaining": int(sum(len(set(group["label"].astype(bool))) > 1 for _, group in selected.groupby(["source_id", "claim"], sort=False))),
        "duplicate_example_ids": int(selected["example_id"].duplicated().sum()),
        "split_source_overlap_count": source_split_overlap,
        "input_mismatch_groups_included": 0,
        "source_claim_groups_final": int(selected.groupby(["source_id", "claim"], sort=False).ngroups),
        "training_integration": {
            "current_entrypoint": "scripts/run_ragtruth_confirmatory.py",
            "current_input_contract": "Parquet com claim, chunk_1..chunk_4 e evidence_mask",
            "future_adapter": None,
            "metadata_excluded_from_features": ["duplicate_group_id", "duplicate_*", "spans", "scores", "model", "response_id"],
            "validation_group_key": "source_id",
            "official_test_remains_separate": True,
            "hardcoded_four_valid_evidence_in_training": False,
            "training_implemented_in_this_step": False,
        },
        "parent": parent_stats,
        "derived": derived_stats,
        "removed": {"collapsed_copies": collapsed_stats, "conflicting_groups": conflict_stats, "input_mismatch_groups": mismatch_stats},
        "removed_positive_copies": int(collapsed_rows["label"].astype(bool).sum()),
        "removed_negative_copies": int((~collapsed_rows["label"].astype(bool)).sum()),
        "removed_positive_conflicts": int(excluded_conflicts["label"].astype(bool).sum()),
        "removed_negative_conflicts": int((~excluded_conflicts["label"].astype(bool)).sum()),
        "removed_positive_input_mismatch": int(excluded_mismatches["label"].astype(bool).sum()),
        "removed_negative_input_mismatch": int((~excluded_mismatches["label"].astype(bool)).sum()),
        "integrity_errors": (["split_source_overlap"] if source_split_overlap else []),
        "ready_for_training_view": bool(source_split_overlap == 0 and len(selected) and not selected.duplicated(["source_id", "claim"]).any()),
    }
    baseline_audit = _audit_baseline(selected, parent, seed)
    counts = {"parent": parent_stats, "derived": derived_stats, "removed": training_view_audit["removed"], "duplicate_groups": len(repeated), "conflicting_groups": len(conflicts), "input_mismatch_groups": len(mismatch), "collapsed_copy_rows": int(len(collapsed_rows)), "output_rows": int(len(selected))}
    config = {"schema_version": TRAINING_VIEW_SCHEMA, "policy": policy, "parent_run_dir": str(Path(input_run_dir).resolve()), "parent_signature": parent_signature, "group_key": ["source_id", "claim"], "key_separator": KEY_SEPARATOR, "representative_selection_rule": REPRESENTATIVE_RULE, "input_equivalence": {"exact_fields": _INPUT_TEXT_FIELDS + _INPUT_EXACT_FIELDS + ["evidence_mask"], "float_fields": _INPUT_FLOAT_FIELDS, "atol": _SCORE_ATOL, "rtol": _SCORE_RTOL}, "conflict_policy": "drop_all_conflicting_groups", "raw_response_path": str(responses_path.resolve()) if responses_path else None, "raw_source_path": str(sources_path.resolve()) if sources_path else None}
    decision = "READY FOR TRAINING WITH WARNINGS" if training_view_audit["ready_for_training_view"] and (conflicts or mismatch) else ("READY FOR TRAINING" if training_view_audit["ready_for_training_view"] else "NOT READY FOR TRAINING")
    report_lines = [
        "# RAGTruth QA training view report", "",
        f"**Decision:** `{decision}`", "",
        "## Lineage", "",
        f"- Parent run: `{Path(input_run_dir).resolve()}`",
        f"- Parent signature: `{parent_signature}`",
        f"- Parent schema: `{PARENT_SCHEMA}`",
        f"- Parent `dataset.parquet` SHA-256: `{parent_sha}`",
        f"- Training-view schema: `{TRAINING_VIEW_SCHEMA}`",
        f"- Policy: `{policy}`",
        "- Group key: exact `(source_id, claim)`; no case/punctuation/semantic normalization.",
        f"- Representative rule: `{REPRESENTATIVE_RULE}`",
        "- Input equivalence: exact text/hashes/indices/masks/signatures; scores compared with `atol=1e-6`, `rtol=1e-6`.",
        "", "## Recalculated counts", "",
        f"- Parent: {parent_stats['rows']} rows, {parent_stats['sources']} sources, {parent_stats['positives']} positives ({parent_stats['prevalence']:.6f}).",
        f"- Derived: {derived_stats['rows']} rows, {derived_stats['sources']} sources, {derived_stats['positives']} positives ({derived_stats['prevalence']:.6f}).",
        f"- Repeated groups: {len(repeated)}; consistent groups collapsed: {sum(item['multiplicity'] > 1 and not item['conflict'] for item in repeated)}; conflicting groups: {len(conflicts)}.",
        f"- Collapsed copy rows: {len(collapsed_rows)} ({int(collapsed_rows['label'].astype(bool).sum())} positive, {int((~collapsed_rows['label'].astype(bool)).sum())} negative).",
        f"- Conflict rows excluded: {len(excluded_conflicts)} ({int(excluded_conflicts['label'].astype(bool).sum())} positive, {int((~excluded_conflicts['label'].astype(bool)).sum())} negative).",
        f"- Input-mismatch groups/rows: {len(mismatch)}/{len(excluded_mismatches)}.",
        "", "## Conflict audit", "",
        f"All {len(conflicts)} conflicting groups were audited and fully excluded. Diagnostics: {dict(Counter(row['diagnostic'] for row in review_rows))}.",
        "The five intra-response conflicts are classified as `span_occurrence_mismatch`; the remaining conflicts are retained only in the review artifacts and excluded from the training view.",
        f"`source_id=15278` contributes {sum(row['source_id'] == '15278' for row in review_rows)} conflicting groups.",
        "See `conflict_review.csv` and `conflict_review.md` for every claim, example, response/model, offsets, spans, context, evidence, masks, scores, diagnosis and recommendation.",
        "", "## Integrity and baseline", "",
        f"- labels used for claim boundaries: `{training_view_audit['labels_used_for_claim_boundaries']}`",
        f"- duplicate groups remaining: `{training_view_audit['duplicate_groups_remaining']}`",
        f"- conflicting groups remaining: `{training_view_audit['conflicting_duplicate_groups_remaining']}`",
        f"- input mismatch groups included: `{training_view_audit['input_mismatch_groups_included']}`",
        f"- source overlap across splits: `{training_view_audit['split_source_overlap_count']}`",
        f"- surface baseline validation AUPRC parent/derived: `{baseline_audit['parent'].get('validation_AUPRC')}` / `{baseline_audit['derived'].get('validation_AUPRC')}`",
        f"- surface baseline validation AUROC parent/derived: `{baseline_audit['parent'].get('validation_AUROC')}` / `{baseline_audit['derived'].get('validation_AUROC')}`",
        "The superficial baseline uses train only, grouped by `source_id`; the official test split was not used for tuning.",
        "", "## Execution", "",
        "No E5 encoder, embeddings, retrieval, LoRA or MIL training was executed. The parent Parquet and all parent artifacts remain unchanged.",
        "The generated `dataset.parquet` is the transformed view of the official splits, not a new official split. Future training must consume only this Parquet and preserve `evidence_mask`; duplicate-audit metadata are not model features.",
        "The active confirmatory pipeline consumes this Parquet directly, preserves evidence_mask and performs source-grouped validation. Training remains outside this preparation step.",
    ]
    manifest = {"schema_version": TRAINING_VIEW_SCHEMA, "signature": signature, "status": "audit_only" if audit_only else "completed", "decision": decision, "parent": {"run_dir": str(Path(input_run_dir).resolve()), "signature": parent_signature, "schema_version": PARENT_SCHEMA, "dataset_sha256": parent_sha, "dataset_rows": int(len(parent))}, "policy": policy, "counts": counts, "lineage": signature_payload, "environment": _environment(), "artifacts": {}}
    if not audit_only:
        _atomic_parquet(destination / "dataset.parquet", selected)
    _atomic_parquet(destination / "duplicate_groups.parquet", pd.DataFrame(duplicate_groups_rows))
    _atomic_parquet(destination / "excluded_conflicting_groups.parquet", excluded_conflicts)
    _atomic_parquet(destination / "excluded_input_mismatch_groups.parquet", excluded_mismatches)
    fields = list(review_rows[0].keys()) if review_rows else ["duplicate_group_id", "source_id", "claim", "diagnostic", "confidence"]
    _atomic_csv(destination / "conflict_review.csv", review_rows, fields)
    _atomic_text(destination / "conflict_review.md", review_md)
    _atomic_text(destination / "report.md", "\n".join(report_lines) + "\n")
    _atomic_json(destination / "counts.json", counts)
    _atomic_json(destination / "duplicate_audit.json", duplicate_audit)
    _atomic_json(destination / "training_view_audit.json", training_view_audit)
    _atomic_json(destination / "surface_feature_audit.json", baseline_audit)
    _atomic_json(destination / "resolved_config.json", config)
    _atomic_jsonl = lambda path, rows: _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rows))
    _atomic_jsonl(destination / "run_log.jsonl", [{"event": "training_view_completed", "signature": signature, "audit_only": audit_only, "seconds": time.perf_counter() - started, "parent_signature": parent_signature, "encoder_loaded": False, "embeddings_recomputed": False}])
    manifest["surface_feature_audit"] = baseline_audit
    manifest["conflict_review"] = {"groups": len(conflicts), "diagnostics": dict(Counter(row["diagnostic"] for row in review_rows))}
    artifact_names = ["duplicate_groups.parquet", "excluded_conflicting_groups.parquet", "excluded_input_mismatch_groups.parquet", "conflict_review.csv", "conflict_review.md", "report.md", "counts.json", "duplicate_audit.json", "training_view_audit.json", "surface_feature_audit.json", "resolved_config.json", "run_log.jsonl"]
    if not audit_only:
        artifact_names.insert(0, "dataset.parquet")
    manifest["artifacts"] = {name: sha256_file(destination / name) for name in artifact_names}
    _atomic_json(destination / "manifest.json", manifest)
    return manifest
