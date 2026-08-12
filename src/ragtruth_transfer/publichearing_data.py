from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_jsonl


def build_publichearing_examples(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(path)
    examples: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    for record in records:
        hearing_id = str(record["id"])
        metadata = record["metadados_extraidos"]
        for person_index, person in enumerate(metadata.get("envolvidos", [])):
            for opinion_index, opinion_entry in enumerate(person.get("opinioes", [])):
                verification = opinion_entry.get("verificacao_alucinacao", {})
                chunks = [str(value) for value in (opinion_entry.get("chunks_proximos") or [])]
                automatic_predictions = {
                    key: value.get("alucinacao")
                    for key, value in verification.items()
                    if key.startswith("prompt_") and isinstance(value, dict)
                }
                row = {
                    "sample_id": f"{hearing_id}:{person_index}:{opinion_index}",
                    "hearing_id": hearing_id,
                    "subject": str(metadata.get("assunto", "")),
                    "person_name": str(person.get("nome", "")),
                    "person_role": str(person.get("cargo", "")),
                    "opinion": str(opinion_entry.get("opiniao", "")),
                    "context_chunks": chunks,
                    "chunk_count": len(chunks),
                    "manual_hallucination": bool(verification["verificacao_manual"]),
                    **automatic_predictions,
                }
                full_rows.append(row)
                if len(chunks) == 4:
                    examples.append(
                        {
                            "example_id": row["sample_id"],
                            "source_id": hearing_id,
                            "claim": row["opinion"],
                            "label": row["manual_hallucination"],
                            "evidence": chunks,
                            "evidence_mask": [True, True, True, True],
                            "task_type": "PublicHearingBR",
                        }
                    )
    return examples, full_rows


def export_publichearing_examples(path: Path, output_path: Path) -> list[dict[str, Any]]:
    examples, full_rows = build_publichearing_examples(path)
    write_jsonl(output_path, examples)
    return full_rows
