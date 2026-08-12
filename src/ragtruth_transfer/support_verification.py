
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io_utils import iter_jsonl, read_jsonl, sha256_file, write_json, write_jsonl

SPLITS = ("train", "validation", "test")
DEFAULT_SEPARATOR = "\n\n[SEP]\n\n"
SCORE_FILENAME = "support_verification.jsonl"
PARTIAL_FILENAME = "support_verification.jsonl.partial"
MODE_VERSION = "support-verification-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    return _sha256_text(_canonical(value))


def _normalise_label(label: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _valid_evidence(row: dict[str, Any]) -> tuple[list[str], list[bool]]:
    evidence = row.get("evidence")
    mask = row.get("evidence_mask")
    if not isinstance(evidence, list) or not isinstance(mask, list) or len(evidence) != len(mask):
        raise ValueError(f"evidence/evidence_mask inválidos para {row.get('example_id')!r}")
    valid = [str(text) for text, present in zip(evidence, mask) if bool(present) and str(text).strip()]
    return valid, [bool(value) for value in mask]


def _broad_paths(broad_dir: Path) -> list[Path]:
    paths = [broad_dir / f"{split}.jsonl" for split in SPLITS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Arquivos Broad ausentes: " + ", ".join(missing))
    return paths


def _load_broad(broad_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for path in _broad_paths(broad_dir):
        split = path.stem
        hashes[split] = sha256_file(path)
        for row in iter_jsonl(path):
            row = dict(row)
            row["_broad_split"] = split
            if "split" in row and str(row["split"]) != split:
                raise ValueError(f"Split divergente em {path}: {row.get('example_id')}")
            example_id = str(row.get("example_id", ""))
            if not example_id or example_id in seen:
                raise ValueError(f"example_id ausente ou duplicado: {example_id!r}")
            seen.add(example_id)
            _valid_evidence(row)
            rows.append(row)
    return rows, hashes


def tokenizer_fingerprint(tokenizer: Any) -> str:
    payload: dict[str, Any] = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "init_kwargs": getattr(tokenizer, "init_kwargs", {}),
    }
    try:
        payload["special_tokens_map"] = tokenizer.special_tokens_map
    except Exception:
        payload["special_tokens_map"] = {}
    return _hash_json(payload)


def _tokenizer_fingerprint_from_id(tokenizer_id: str, revision: str | None) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)
    return tokenizer_fingerprint(tokenizer)


def _execution_fingerprint(
    broad_hashes: dict[str, str],
    verifier_model: str,
    verifier_revision: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    tokenizer_fp: str,
    max_length: int,
    separator: str,
    max_negative_examples_per_split: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode_version": MODE_VERSION,
        "broad_sha256": broad_hashes,
        "verifier_model_id": verifier_model,
        "verifier_revision": verifier_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_fingerprint": tokenizer_fp,
        "max_length": int(max_length),
        "separator": separator,
        "max_negative_examples_per_split": max_negative_examples_per_split,
    }
    payload["fingerprint_sha256"] = _hash_json(payload)
    return payload


class EntailmentVerifier:

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.tokenizer_id = tokenizer_id or model_id
        self.tokenizer_revision = tokenizer_revision if tokenizer_revision is not None else revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, revision=self.tokenizer_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.entailment_index = self._find_entailment_index()

    def _find_entailment_index(self) -> int:
        mapping = getattr(self.model.config, "id2label", None) or {}
        labels = {int(index): str(label) for index, label in mapping.items()}
        matches = [
            index
            for index, label in labels.items()
            if _normalise_label(label) in {"entailment", "entail"}
            or "entailment" in _normalise_label(label)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Mapeamento de entailment ambíguo ou ausente; "
                f"id2label={mapping!r}, matches={matches!r}"
            )
        return matches[0]

    def score_pairs(self, premises: Sequence[str], claims: Sequence[str], batch_size: int) -> list[float]:
        if len(premises) != len(claims):
            raise ValueError("premises e claims devem ter o mesmo tamanho")
        scores: list[float] = []
        for start in range(0, len(premises), batch_size):
            batch_premises = list(premises[start : start + batch_size])
            batch_claims = list(claims[start : start + batch_size])
            tensors = self.tokenizer(
                batch_premises,
                batch_claims,
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            tensors = {key: value.to(self.device) for key, value in tensors.items()}
            with torch.inference_mode():
                logits = self.model(**tensors).logits
                probabilities = torch.softmax(logits, dim=-1)[:, self.entailment_index]
            scores.extend(float(value) for value in probabilities.detach().cpu())
        return scores

    max_length: int = 512


def _pair_token_counts(tokenizer: Any, premise: str, claim: str, max_length: int) -> tuple[int, int, bool]:
    before = tokenizer(premise, claim, truncation=False, add_special_tokens=True)["input_ids"]
    after = tokenizer(
        premise,
        claim,
        truncation="only_first",
        max_length=max_length,
        add_special_tokens=True,
    )["input_ids"]
    before_count = len(before)
    after_count = len(after)
    return before_count, after_count, after_count < before_count


def score_row(
    row: dict[str, Any],
    verifier: EntailmentVerifier,
    batch_size: int,
    max_length: int,
    separator: str,
    execution_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    if bool(row.get("label")):
        raise ValueError("score deve processar somente exemplos negativos; positivo recebido.")
    verifier.max_length = max_length
    valid, evidence_mask = _valid_evidence(row)
    claim = str(row.get("claim", ""))
    individual_scores = [None] * len(evidence_mask)
    if valid:
        valid_scores = verifier.score_pairs(valid, [claim] * len(valid), batch_size)
        cursor = 0
        for index, present in enumerate(evidence_mask):
            if present and str(row["evidence"][index]).strip():
                individual_scores[index] = valid_scores[cursor]
                cursor += 1
    finite_scores = [value for value in individual_scores if value is not None]
    best_index = None
    individual_max = None
    if finite_scores:
        best_index = max(
            (index for index, value in enumerate(individual_scores) if value is not None),
            key=lambda index: (float(individual_scores[index]), -index),
        )
        individual_max = float(individual_scores[best_index])

    concat = separator.join(valid)
    concat_before, concat_after, concat_truncated = _pair_token_counts(
        verifier.tokenizer, concat, claim, max_length
    ) if valid else (0, 0, False)
    concat_score = verifier.score_pairs([concat], [claim], batch_size)[0] if valid else None
    record = {
        "split": str(row["_broad_split"]),
        "example_id": str(row["example_id"]),
        "source_id": str(row.get("source_id", "")),
        "label": bool(row.get("label", False)),
        "claim_sha256": _sha256_text(claim),
        "evidence_sha256": _hash_json(row.get("evidence", [])),
        "evidence_mask": evidence_mask,
        "valid_evidence_count": len(valid),
        "individual_scores": individual_scores,
        "individual_max_score": individual_max,
        "best_evidence_index": best_index,
        "concat_score": concat_score,
        "concat_tokens_before": concat_before,
        "concat_tokens_after": concat_after,
        "concat_truncated": concat_truncated,
        "concat_separator": separator,
        "verifier_model_id": verifier.model_id,
        "verifier_revision": verifier.revision,
        "tokenizer_id": verifier.tokenizer_id,
        "tokenizer_revision": verifier.tokenizer_revision,
        "tokenizer_fingerprint": execution_fingerprint["tokenizer_fingerprint"],
        "max_length": max_length,
        "execution_fingerprint": execution_fingerprint,
    }
    return record


def _atomic_jsonl_replace(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _load_score_records(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _select_negative_rows(
    rows: list[dict[str, Any]], max_negative_examples_per_split: int | None
) -> list[dict[str, Any]]:
    negative_rows = [row for row in rows if not bool(row.get("label"))]
    if max_negative_examples_per_split is None:
        return negative_rows
    if max_negative_examples_per_split <= 0:
        raise ValueError("max_negative_examples_per_split deve ser positivo.")
    selected_by_split: dict[str, int] = Counter()
    selected: list[dict[str, Any]] = []
    for row in negative_rows:
        split = str(row["_broad_split"])
        if selected_by_split[split] < max_negative_examples_per_split:
            selected.append(row)
            selected_by_split[split] += 1
    return selected


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    broad_dir = Path(args.broad_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / SCORE_FILENAME
    partial_path = output_dir / PARTIAL_FILENAME
    if final_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Já existe {final_path}; use --resume ou --overwrite.")
    if args.overwrite:
        for path in (final_path, partial_path):
            if path.exists():
                path.unlink()

    rows, broad_hashes = _load_broad(broad_dir)
    max_negative_examples_per_split = getattr(args, "max_negative_examples_per_split", None)
    negative_rows = _select_negative_rows(rows, max_negative_examples_per_split)
    tokenizer_id = args.tokenizer_id or args.verifier_model
    tokenizer_revision = args.tokenizer_revision if args.tokenizer_revision is not None else args.verifier_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
    tokenizer_fp = tokenizer_fingerprint(tokenizer)
    execution_fp = _execution_fingerprint(
        broad_hashes,
        args.verifier_model,
        args.verifier_revision,
        tokenizer_id,
        tokenizer_revision,
        tokenizer_fp,
        args.max_length,
        args.separator,
        max_negative_examples_per_split,
    )

    existing_path = partial_path if partial_path.is_file() else (final_path if args.resume and final_path.is_file() else None)
    existing = _load_score_records(existing_path) if existing_path is not None else []
    by_id: dict[str, dict[str, Any]] = {}
    for record in existing:
        example_id = str(record.get("example_id", ""))
        if not example_id or example_id in by_id:
            raise ValueError(f"Registro parcial duplicado ou inválido: {example_id!r}")
        if record.get("execution_fingerprint") != execution_fp:
            raise ValueError("Fingerprint incompatível: não é seguro reutilizar o partial.")
        by_id[example_id] = record

    # Normalize the working file into Broad order before appending new records.
    if existing:
        _atomic_jsonl_replace(
            partial_path,
            (by_id[str(row["example_id"])] for row in negative_rows if str(row["example_id"]) in by_id),
        )
    verifier: EntailmentVerifier | None = None
    expected_ids = {str(row["example_id"]) for row in negative_rows}
    unexpected_ids = set(by_id) - expected_ids
    if unexpected_ids:
        raise ValueError("Partial contém positivos ou exemplos fora do limite de smoke.")
    pending = [row for row in negative_rows if str(row["example_id"]) not in by_id]
    if pending:
        verifier = EntailmentVerifier(
            args.verifier_model,
            args.verifier_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            device=args.device,
        )
        verifier.max_length = args.max_length
        # Keep the tokenizer fingerprint tied to the actual loaded verifier tokenizer.
        if tokenizer_fingerprint(verifier.tokenizer) != tokenizer_fp:
            raise RuntimeError("Tokenizer mudou entre a preparação e o carregamento do verificador.")
        with partial_path.open("a", encoding="utf-8") as handle:
            for row in tqdm(pending, desc="Support scores"):
                record = score_row(
                    row,
                    verifier,
                    args.batch_size,
                    args.max_length,
                    args.separator,
                    execution_fp,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    final_records = _load_score_records(partial_path) if partial_path.is_file() else existing
    if len(final_records) != len(negative_rows) or {str(item["example_id"]) for item in final_records} != expected_ids:
        raise RuntimeError("Scores incompletos; o partial foi preservado para retomada.")
    _atomic_jsonl_replace(final_path, final_records)
    if partial_path.exists():
        partial_path.unlink()
    manifest = {
        "protocol": "support scoring only; no final Strict decision",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "broad_dir": str(broad_dir.resolve()),
        "broad_sha256": broad_hashes,
        "scores_path": str(final_path.resolve()),
        "scores_sha256": sha256_file(final_path),
        "broad_examples": len(rows),
        "negative_examples": len(negative_rows),
        "negative_examples_by_split": dict(Counter(str(row["_broad_split"]) for row in negative_rows)),
        "fingerprint": execution_fp,
        "verifier_model_id": args.verifier_model,
        "verifier_revision": args.verifier_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "separator": args.separator,
        "device": str(getattr(verifier, "device", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))),
        "pairs": {
            "individual": int(sum(sum(value is not None for value in item.get("individual_scores", [])) for item in final_records)),
            "concat": int(sum(item.get("concat_score") is not None for item in final_records)),
        },
        "concat_truncated": int(sum(bool(item.get("concat_truncated")) for item in final_records)),
        "score_distributions": {
            "individual_max_score": _score_distribution([float(item["individual_max_score"]) for item in final_records if item.get("individual_max_score") is not None]),
            "concat_score": _score_distribution([float(item["concat_score"]) for item in final_records if item.get("concat_score") is not None]),
        },
        "cuda_max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(getattr(verifier, "device", torch.device("cuda")))
        ) if verifier is not None and torch.cuda.is_available() else None,
    }
    write_json(output_dir / "support_verification_manifest.json", manifest)
    return manifest


def _score_map(scores_path: Path) -> dict[str, dict[str, Any]]:
    records = _load_score_records(scores_path)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("example_id", ""))
        if not key or key in result:
            raise ValueError(f"Score duplicado ou inválido: {key!r}")
        result[key] = record
    return result


def _quantile_edges(finite_values: list[float], bins: int = 4) -> np.ndarray:
    if not finite_values or len(set(finite_values)) < 2:
        return np.asarray([], dtype=float)
    return np.quantile(np.asarray(finite_values, dtype=float), np.linspace(0, 1, bins + 1))[1:-1]


def _quantile_bin(value: float | None, edges: np.ndarray) -> str:
    if value is None:
        return "missing"
    if edges.size == 0:
        return "q0"
    return f"q{min(edges.size, int(np.searchsorted(edges, value, side='right')))}"


def _audit_stratum(
    row: dict[str, Any],
    score: dict[str, Any],
    individual_edges: np.ndarray,
    concat_edges: np.ndarray,
) -> tuple[Any, ...]:
    individual = _safe_float(score.get("individual_max_score"))
    concat = _safe_float(score.get("concat_score"))
    individual_label = _quantile_bin(individual, individual_edges)
    concat_label = _quantile_bin(concat, concat_edges)
    agree = "agree" if individual_label == concat_label and individual_label != "missing" else "disagree"
    return (
        str(row["_broad_split"]),
        str(row.get("task_type", "unknown")),
        individual_label,
        concat_label,
        agree,
        int(score.get("valid_evidence_count", 0)),
        "truncated" if bool(score.get("concat_truncated")) else "not_truncated",
    )


def _stratified_indices(strata: list[tuple[Any, ...]], sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0:
        raise ValueError("sample_size deve ser positivo")
    if sample_size >= len(strata):
        return list(range(len(strata)))
    rng = np.random.default_rng(seed)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(strata):
        groups[key].append(index)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    keys = sorted(groups, key=str)
    # First cover each stratum, then allocate remaining slots proportionally.
    for key in keys:
        if len(selected) >= sample_size:
            break
        selected.append(groups[key].pop())
    remaining = sample_size - len(selected)
    pool = [index for values in groups.values() for index in values]
    if remaining > 0 and pool:
        selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return sorted(selected)


AUDIT_FIELDS = [
    "split", "example_id", "source_id", "response_id", "task_type", "claim",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4", "evidence_mask",
    "individual_scores", "individual_max_score", "best_evidence_index", "concat_score",
    "concat_truncated", "valid_evidence_count", "manual_decision", "comment",
]


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows, _ = _load_broad(Path(args.broad_dir))
    score_map = _score_map(Path(args.scores_path))
    selected_rows = [
        row
        for row in rows
        if not bool(row.get("label"))
        and (args.include_test or str(row["_broad_split"]) in {"train", "validation"})
    ]
    missing = [str(row["example_id"]) for row in selected_rows if str(row["example_id"]) not in score_map]
    if missing:
        raise ValueError(f"Scores ausentes para {len(missing)} exemplos; primeiro={missing[0]}")
    eligible_ids = {str(row["example_id"]) for row in selected_rows}
    all_score_records = [score_map[example_id] for example_id in eligible_ids]
    individual_values = [
        value for value in (_safe_float(item.get("individual_max_score")) for item in all_score_records)
        if value is not None
    ]
    concat_values = [
        value for value in (_safe_float(item.get("concat_score")) for item in all_score_records)
        if value is not None
    ]
    individual_edges = _quantile_edges(individual_values)
    concat_edges = _quantile_edges(concat_values)
    strata = [
        _audit_stratum(row, score_map[str(row["example_id"])], individual_edges, concat_edges)
        for row in selected_rows
    ]
    indices = _stratified_indices(strata, min(args.sample_size, len(selected_rows)), args.seed)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for index in indices:
            row = selected_rows[index]
            score = score_map[str(row["example_id"])]
            evidence = list(row.get("evidence", []))
            values = {
                "split": row["_broad_split"],
                "example_id": row["example_id"],
                "source_id": row.get("source_id", ""),
                "response_id": row.get("response_id", ""),
                "task_type": row.get("task_type", ""),
                "claim": row.get("claim", ""),
                **{f"evidence_{i + 1}": evidence[i] if i < len(evidence) else "" for i in range(4)},
                "evidence_mask": json.dumps(row.get("evidence_mask", []), ensure_ascii=False),
                "individual_scores": json.dumps(score.get("individual_scores", []), ensure_ascii=False),
                "individual_max_score": score.get("individual_max_score", ""),
                "best_evidence_index": score.get("best_evidence_index", ""),
                "concat_score": score.get("concat_score", ""),
                "concat_truncated": score.get("concat_truncated", False),
                "valid_evidence_count": score.get("valid_evidence_count", 0),
                "manual_decision": "",
                "comment": "",
            }
            writer.writerow(values)
    return {
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "sample_size": len(indices),
        "eligible_examples": len(selected_rows),
        "include_test": bool(args.include_test),
        "seed": args.seed,
        "strata": len(set(strata)),
    }


def _dependency_versions() -> dict[str, str]:
    names = ("numpy", "torch", "transformers", "huggingface-hub", "pytest")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _score_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    broad_dir = Path(args.broad_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Diretório não vazio: {output_dir}; use --overwrite explicitamente.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, broad_hashes = _load_broad(broad_dir)
    score_map = _score_map(Path(args.scores_path))
    negative_ids = {str(row["example_id"]) for row in rows if not bool(row.get("label"))}
    if negative_ids != set(score_map):
        raise ValueError("Scores não correspondem exatamente aos exemplos negativos Broad.")

    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if source_id:
            source_splits[source_id].add(str(row["_broad_split"]))
    overlapping_sources = sorted(source_id for source_id, splits in source_splits.items() if len(splits) > 1)
    if overlapping_sources:
        raise ValueError(
            "source_id aparece em mais de um split Broad; materialização insegura: "
            + ", ".join(overlapping_sources[:5])
        )

    outputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    before = Counter(str(row["_broad_split"]) for row in rows)
    positives_before = Counter(str(row["_broad_split"]) for row in rows if bool(row.get("label")))
    kept_negative = Counter()
    removed_negative = Counter()
    for row in rows:
        is_positive = bool(row.get("label"))
        if is_positive:
            score = None
        else:
            score = score_map[str(row["example_id"])]
        if is_positive:
            keep = True
        else:
            individual_score = _safe_float(score.get("individual_max_score"))
            individual_keep = individual_score is not None and individual_score >= args.individual_threshold
            concat_keep = False
            if args.allow_nontruncated_concat and not bool(score.get("concat_truncated")):
                concat_score = _safe_float(score.get("concat_score"))
                concat_keep = concat_score is not None and concat_score >= args.concat_threshold
            keep = individual_keep or concat_keep
        split = str(row["_broad_split"])
        if keep:
            outputs[split].append({key: value for key, value in row.items() if key != "_broad_split"})
            if not is_positive:
                kept_negative[split] += 1
        elif not is_positive:
            removed_negative[split] += 1

    output_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, outputs[split])
        output_hashes[split] = sha256_file(path)

    def _stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        task_counts = Counter(str(item.get("task_type", "unknown")) for item in items)
        return {
            "examples": len(items),
            "positives": sum(bool(item.get("label")) for item in items),
            "positive_rate": (sum(bool(item.get("label")) for item in items) / len(items)) if items else 0.0,
            "tasks": dict(sorted(task_counts.items())),
        }

    score_values_individual = [_safe_float(item.get("individual_max_score")) for item in score_map.values()]
    score_values_concat = [_safe_float(item.get("concat_score")) for item in score_map.values()]
    source_ids_by_split = {split: [str(row.get("source_id", "")) for row in values] for split, values in outputs.items()}
    all_output_ids = [str(row["example_id"]) for values in outputs.values() for row in values]
    broad_ids_by_split = {
        split: [str(row["example_id"]) for row in rows if str(row["_broad_split"]) == split]
        for split in SPLITS
    }
    output_ids_by_split = {
        split: [str(row["example_id"]) for row in outputs[split]]
        for split in SPLITS
    }
    order_checks: dict[str, bool] = {}
    for split in SPLITS:
        broad_iterator = iter(broad_ids_by_split[split])
        order_checks[split] = all(
            any(candidate == expected for candidate in broad_iterator)
            for expected in output_ids_by_split[split]
        )
    manifest = {
        "protocol": "Strict derived from Broad JSONL with explicit thresholds",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "broad_dir": str(broad_dir.resolve()),
        "broad_sha256": broad_hashes,
        "scores_path": str(Path(args.scores_path).resolve()),
        "scores_sha256": sha256_file(Path(args.scores_path)),
        "outputs": output_hashes,
        "thresholds": {
            "individual_max": args.individual_threshold,
            "concat": args.concat_threshold,
        },
        "allow_nontruncated_concat": bool(args.allow_nontruncated_concat),
        "decision_rule": (
            "preserve positives; keep negatives when individual_max_score >= individual_threshold "
            "or, when explicitly enabled, concat_score >= concat_threshold and concat_truncated=false"
        ),
        "counts_before": {split: before[split] for split in SPLITS},
        "counts_after": {split: len(outputs[split]) for split in SPLITS},
        "positives_preserved": {split: positives_before[split] == sum(bool(item.get("label")) for item in outputs[split]) for split in SPLITS},
        "negatives_kept": dict(kept_negative),
        "negatives_removed": dict(removed_negative),
        "stats_by_split": {split: _stats(outputs[split]) for split in SPLITS},
        "score_distributions": {
            "individual_max_score": _score_distribution([v for v in score_values_individual if v is not None]),
            "concat_score": _score_distribution([v for v in score_values_concat if v is not None]),
        },
        "examples_by_valid_evidence_count": dict(Counter(int(item.get("valid_evidence_count", 0)) for item in score_map.values())),
        "source_id_sha256_by_split": {split: _hash_json(sorted(values)) for split, values in source_ids_by_split.items()},
        "example_id_sha256": _hash_json(all_output_ids),
        "order_preserved": order_checks,
        "dependencies": _dependency_versions(),
        "verifier": {
            "model_id": next(iter(score_map.values())).get("verifier_model_id") if score_map else None,
            "revision": next(iter(score_map.values())).get("verifier_revision") if score_map else None,
            "tokenizer_fingerprint": next(iter(score_map.values())).get("tokenizer_fingerprint") if score_map else None,
            "max_length": next(iter(score_map.values())).get("max_length") if score_map else None,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    score = subparsers.add_parser("score", help="calcula scores sem aplicar filtro")
    score.add_argument("--broad-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--verifier-model", required=True)
    score.add_argument("--verifier-revision", required=True)
    score.add_argument("--tokenizer-id")
    score.add_argument("--tokenizer-revision")
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--max-length", type=int, default=512)
    score.add_argument("--separator", default=DEFAULT_SEPARATOR)
    score.add_argument("--device")
    score.add_argument("--max-negative-examples-per-split", type=int)
    score.add_argument("--resume", action="store_true")
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(handler=run_score)

    audit = subparsers.add_parser("audit", help="gera amostra estratificada para auditoria humana")
    audit.add_argument("--broad-dir", type=Path, required=True)
    audit.add_argument("--scores-path", type=Path, required=True)
    audit.add_argument("--output-path", type=Path, required=True)
    audit.add_argument("--sample-size", type=int, default=500)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--include-test", action="store_true")
    audit.set_defaults(handler=run_audit)

    materialize = subparsers.add_parser("materialize", help="materializa Strict com thresholds explícitos")
    materialize.add_argument("--broad-dir", type=Path, required=True)
    materialize.add_argument("--scores-path", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--individual-threshold", type=float, required=True)
    materialize.add_argument("--concat-threshold", type=float, required=True)
    materialize.add_argument("--allow-nontruncated-concat", action="store_true")
    materialize.add_argument("--overwrite", action="store_true")
    materialize.set_defaults(handler=run_materialize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0 or getattr(args, "max_length", 1) <= 0:
        parser.error("batch-size e max-length devem ser positivos")
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
