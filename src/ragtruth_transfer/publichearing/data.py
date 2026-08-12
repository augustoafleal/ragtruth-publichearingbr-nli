from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from ..io_utils import read_jsonl, sha256_file
from .config import PublicHearingConfig


def download_dataset(config: PublicHearingConfig) -> Path:
    return Path(hf_hub_download(
        repo_id=config.dataset_repo, filename=config.dataset_filename,
        repo_type="dataset", revision=config.dataset_revision,
    ))


def normalize_publichearing(path: Path, validate_expected: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record_index, record in enumerate(read_jsonl(path)):
        hearing_id = str(record.get("id", "")).strip()
        metadata = record.get("metadados_extraidos") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        for person_index, person in enumerate(metadata.get("envolvidos") or []):
            if not isinstance(person, dict):
                continue
            for opinion_index, entry in enumerate(person.get("opinioes") or []):
                if not isinstance(entry, dict):
                    continue
                verification = entry.get("verificacao_alucinacao") or {}
                verification = verification if isinstance(verification, dict) else {}
                opinion = str(entry.get("opiniao", "")).strip()
                raw_chunks = entry.get("chunks_proximos") or []
                chunks = [str(chunk).strip() for chunk in raw_chunks] if isinstance(raw_chunks, list) else []
                sample_id = f"{hearing_id}:{person_index}:{opinion_index}"
                base = {
                    "sample_id": sample_id, "hearing_id": hearing_id,
                    "subject": str(metadata.get("assunto", "")).strip(),
                    "person_name": str(person.get("nome", "")).strip(),
                    "person_role": str(person.get("cargo", "")).strip(),
                    "opinion": opinion, "context_chunks": chunks, "chunk_count": len(chunks),
                    "record_index": record_index,
                }
                reasons: list[str] = []
                if not hearing_id:
                    reasons.append("missing_hearing_id")
                if not opinion:
                    reasons.append("empty_opinion")
                if len(chunks) != 4:
                    reasons.append(f"chunk_count_{len(chunks)}")
                elif any(not chunk for chunk in chunks):
                    reasons.append("empty_chunk")
                label = verification.get("verificacao_manual")
                if label is None:
                    reasons.append("missing_manual_label")
                if reasons:
                    rejected.append({**base, "rejection_reason": "|".join(reasons)})
                else:
                    accepted.append({**base, "label": int(bool(label))})
    frame, rejected_frame = pd.DataFrame(accepted), pd.DataFrame(rejected)
    if frame.empty:
        raise RuntimeError("Nenhum exemplo modelável foi encontrado.")
    if not frame.sample_id.is_unique:
        raise RuntimeError("sample_id não é único.")
    audit = {
        "dataset_sha256": sha256_file(path), "records": len(read_jsonl(path)),
        "modelable_examples": len(frame), "hearings": int(frame.hearing_id.nunique()),
        "positives": int(frame.label.sum()), "prevalence": float(frame.label.mean()), "rejected": len(rejected_frame),
    }
    if validate_expected and (audit["hearings"] < 190 or not 4_150 <= audit["modelable_examples"] <= 4_300 or not 0.09 <= audit["prevalence"] <= 0.15):
        raise RuntimeError(f"Contagens inesperadas para PublicHearingBR: {audit}")
    return frame.reset_index(drop=True), rejected_frame.reset_index(drop=True), audit


def write_metadata_csv(frame: pd.DataFrame, rejected: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = frame.copy()
    clean["context_chunks"] = clean["context_chunks"].map(lambda value: json.dumps(value, ensure_ascii=False))
    clean.to_csv(output_dir / "normalized_metadata.csv", index=False)
    rejected_clean = rejected.copy()
    if "context_chunks" in rejected_clean:
        rejected_clean["context_chunks"] = rejected_clean["context_chunks"].map(lambda value: json.dumps(value, ensure_ascii=False))
    rejected_clean.to_csv(output_dir / "rejected_examples.csv", index=False)
