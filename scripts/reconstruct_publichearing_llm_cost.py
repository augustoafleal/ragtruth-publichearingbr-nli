#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any


EXPECTED_EXAMPLES = 4235
PAPER_EXAMPLES = 4238
EXPECTED_PROMPTS = ("prompt_1", "prompt_2", "prompt_3")
MODEL_IDENTIFIERS = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "deepseek": "deepseek-chat",
    "sabia-3.1": "sabia-3.1-2025-05-08",
}
MODEL_ORDER = tuple(MODEL_IDENTIFIERS)
PAIR_KEY_IDENTIFIERS = tuple(MODEL_IDENTIFIERS.values())
OUTPUT_FIELDS = ("alucinacao", "explicacao", "trechos_para_basear_analise")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def load_templates(prompt_dir: Path) -> dict[str, dict[str, str]]:
    templates: dict[str, dict[str, str]] = {}
    for prompt_id in EXPECTED_PROMPTS:
        path = prompt_dir / f"{prompt_id}.txt"
        text = path.read_text(encoding="utf-8")
        marker = "[User]\n"
        if not text.startswith("[System]\n") or marker not in text:
            raise ValueError(f"Invalid prompt template structure: {path}")
        system, user = text[len("[System]\n") :].split(marker, 1)
        if "{TEXTO}" not in user or "{OPINIAO}" not in user:
            raise ValueError(f"Missing placeholders in prompt template: {path}")
        templates[prompt_id] = {"system": system, "user_template": user}
    return templates


def comparison_population_ids(nli_path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ids: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    with nli_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            hearing_id = str(record["id"]).strip()
            metadata = record.get("metadados_extraidos") or {}
            for person_index, person in enumerate(metadata.get("envolvidos") or []):
                for opinion_index, opinion_entry in enumerate(person.get("opinioes") or []):
                    example_id = f"{hearing_id}:{person_index}:{opinion_index}"
                    opinion = str(opinion_entry.get("opiniao", ""))
                    chunks = [str(value) for value in (opinion_entry.get("chunks_proximos") or [])]
                    modelable = bool(opinion.strip()) and len(chunks) == 4 and all(chunk.strip() for chunk in chunks)
                    if not modelable:
                        continue
                    if example_id in records:
                        raise ValueError(f"Duplicate modelable example id: {example_id}")
                    ids.append(example_id)
                    records[example_id] = {
                        "verification": opinion_entry.get("verificacao_alucinacao") or {},
                    }
    if len(ids) != EXPECTED_EXAMPLES:
        raise ValueError(f"Expected {EXPECTED_EXAMPLES} modelable examples, found {len(ids)}")
    return ids, records


def load_original_inputs(csv_path: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            example_id = str(row["sample_id"])
            if example_id in rows:
                raise ValueError(f"Duplicate CSV example id: {example_id}")
            chunks = json.loads(row["context_chunks"])
            if int(row["chunk_count"]) != 4 or len(chunks) != 4 or not all(str(chunk).strip() for chunk in chunks):
                raise ValueError(f"Invalid four-passage input: {example_id}")
            rows[example_id] = {
                "opinion": str(row["opinion"]),
                "evidence_passages": [str(chunk) for chunk in chunks],
                "hearing_id": str(row["hearing_id"]),
            }
    if set(rows) != set(ids):
        missing = sorted(set(ids) - set(rows))
        extra = sorted(set(rows) - set(ids))
        raise ValueError(f"CSV/NLI population mismatch: missing={missing[:3]}, extra={extra[:3]}")
    return rows


def build_examples(ids: list[str], original: dict[str, dict[str, Any]], stored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for example_id in ids:
        if example_id not in original or example_id not in stored:
            raise ValueError(f"Missing aligned source record: {example_id}")
        example = dict(original[example_id])
        example["example_id"] = example_id
        example["verification"] = stored[example_id]["verification"]
        examples.append(example)
    return examples


def input_records(examples: list[dict[str, Any]], templates: dict[str, dict[str, str]]) -> Iterator[dict[str, Any]]:
    for example in examples:
        evidence = "\n\n".join(example["evidence_passages"])
        for prompt_id in EXPECTED_PROMPTS:
            template = templates[prompt_id]
            user_message = template["user_template"].replace("{TEXTO}", evidence).replace("{OPINIAO}", example["opinion"])
            yield {
                "example_id": example["example_id"],
                "prompt_id": prompt_id,
                "opinion": example["opinion"],
                "evidence_passages": example["evidence_passages"],
                "concatenated_evidence": evidence,
                "system_message": template["system"],
                "user_message": user_message,
            }


def canonical_output(decision: dict[str, Any], prompt_id: str) -> str:
    if not isinstance(decision, dict) or not isinstance(decision.get("alucinacao"), bool):
        raise ValueError(f"Missing boolean stored judgment for {prompt_id}")
    output: dict[str, Any] = {"alucinacao": decision["alucinacao"]}
    if not isinstance(decision.get("explicacao"), str):
        raise ValueError(f"Missing explanation for {prompt_id}")
    output["explicacao"] = decision["explicacao"]
    if prompt_id in {"prompt_2", "prompt_3"}:
        passages = decision.get("trechos_para_basear_analise")
        if not isinstance(passages, list) or not all(isinstance(item, str) for item in passages):
            raise ValueError(f"Missing supporting passages for {prompt_id}")
        output["trechos_para_basear_analise"] = passages
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def output_records(examples: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for example in examples:
        verification = example["verification"]
        for model, identifier in MODEL_IDENTIFIERS.items():
            for prompt_id in EXPECTED_PROMPTS:
                pair_key = f"{prompt_id}_{identifier}"
                if pair_key not in verification:
                    raise ValueError(f"Missing pair key {pair_key} for {example['example_id']}")
                yield {
                    "example_id": example["example_id"],
                    "model": model,
                    "model_identifier": identifier,
                    "prompt_id": prompt_id,
                    "canonical_output": canonical_output(verification[pair_key], prompt_id),
                }


class TokenCounter:
    def __init__(self) -> None:
        self.encodings: dict[str, Any] = {}
        try:
            import tiktoken  # type: ignore
        except ImportError:
            self.tiktoken = None
            return
        self.tiktoken = tiktoken
        for model, identifier in MODEL_IDENTIFIERS.items():
            if not model.startswith("gpt"):
                continue
            try:
                self.encodings[model] = tiktoken.encoding_for_model(identifier)
            except Exception:
                self.encodings[model] = tiktoken.get_encoding("o200k_base")

    def info(self, model: str) -> tuple[str, str, str]:
        if model in self.encodings:
            return "tiktoken encoding_for_model/text-only", "COMPATIBLE TOKENIZER", "Does not include provider chat-wrapper overhead."
        return "NOT AVAILABLE LOCALLY", "NOT AVAILABLE", "No reliable local tokenizer was selected; no token count was invented."

    def count(self, model: str, text: str) -> int | None:
        encoding = self.encodings.get(model)
        return len(encoding.encode(text, disallowed_special=())) if encoding is not None else None


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    handle_context = gzip.GzipFile(filename=str(path), mode="wb", compresslevel=9, mtime=0) if path.suffix == ".gz" else path.open("wb")
    with handle_context as handle:
        for record in records:
            data = json_line(record)
            handle.write(data)
            digest.update(data)
            count += 1
    return digest.hexdigest(), count


def digest_jsonl(records: Iterable[dict[str, Any]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for record in records:
        digest.update(json_line(record))
        count += 1
    return digest.hexdigest(), count


def token_records(examples: list[dict[str, Any]], templates: dict[str, dict[str, str]], counter: TokenCounter) -> Iterator[dict[str, Any]]:
    input_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in input_records(examples, templates):
        input_by_key[(record["example_id"], record["prompt_id"])] = record
    for output in output_records(examples):
        input_record = input_by_key[(output["example_id"], output["prompt_id"])]
        model = output["model"]
        input_tokens = counter.count(model, input_record["system_message"] + "\n" + input_record["user_message"])
        output_tokens = counter.count(model, output["canonical_output"])
        tokenizer_name, tokenizer_classification, tokenizer_limitation = counter.info(model)
        yield {
            "example_id": output["example_id"],
            "model": model,
            "model_identifier": output["model_identifier"],
            "prompt_id": output["prompt_id"],
            "input_tokens": input_tokens,
            "output_proxy_tokens": output_tokens,
            "tokenizer_name": tokenizer_name,
            "tokenizer_classification": tokenizer_classification,
            "tokenizer_limitation": tokenizer_limitation,
            "input_reconstruction_status": "RECONSTRUCTABLE WITH ASSUMPTIONS",
            "output_reconstruction_status": "CANONICAL OUTPUT PROXY",
        }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
            count += 1
    return sha256_file(path), count


def aggregate_token_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["prompt_id"]), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for prompt_id in EXPECTED_PROMPTS:
            group = grouped[(model, prompt_id)]
            input_values = [row["input_tokens"] for row in group if row["input_tokens"] is not None]
            output_values = [row["output_proxy_tokens"] for row in group if row["output_proxy_tokens"] is not None]
            tokenizer_name, tokenizer_classification, tokenizer_limitation = (
                group[0]["tokenizer_name"], group[0]["tokenizer_classification"], group[0]["tokenizer_limitation"]
            )
            aggregate.append(
                {
                    "model": model,
                    "model_identifier": MODEL_IDENTIFIERS[model],
                    "prompt_id": prompt_id,
                    "n_examples": len(group),
                    "total_input_tokens": sum(input_values) if input_values else None,
                    "mean_input_tokens": statistics.mean(input_values) if input_values else None,
                    "median_input_tokens": statistics.median(input_values) if input_values else None,
                    "total_output_proxy_tokens": sum(output_values) if output_values else None,
                    "mean_output_proxy_tokens": statistics.mean(output_values) if output_values else None,
                    "median_output_proxy_tokens": statistics.median(output_values) if output_values else None,
                    "tokenizer_name": tokenizer_name,
                    "tokenizer_classification": tokenizer_classification,
                    "input_reconstruction_status": "RECONSTRUCTABLE WITH ASSUMPTIONS",
                    "output_reconstruction_status": "CANONICAL OUTPUT PROXY",
                    "notes": tokenizer_limitation,
                }
            )
    return aggregate


def validate(examples: list[dict[str, Any]], input_rows: list[dict[str, Any]], output_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(examples) != EXPECTED_EXAMPLES:
        raise ValueError("Validation failed: example count")
    if len(input_rows) != EXPECTED_EXAMPLES * len(EXPECTED_PROMPTS):
        raise ValueError("Validation failed: input rows")
    if len(output_rows) != EXPECTED_EXAMPLES * len(MODEL_ORDER) * len(EXPECTED_PROMPTS):
        raise ValueError("Validation failed: output rows")
    if len(token_rows) != len(output_rows):
        raise ValueError("Validation failed: token rows")
    input_keys = {(row["example_id"], row["prompt_id"]) for row in input_rows}
    if len(input_keys) != len(input_rows):
        raise ValueError("Validation failed: duplicate input key")
    output_keys = {(row["example_id"], row["model"], row["prompt_id"]) for row in output_rows}
    if len(output_keys) != len(output_rows):
        raise ValueError("Validation failed: duplicate output key")
    if any(len(row["evidence_passages"]) != 4 for row in input_rows):
        raise ValueError("Validation failed: not all inputs have four passages")
    return {
        "examples": len(examples),
        "prompts": len(EXPECTED_PROMPTS),
        "models": len(MODEL_ORDER),
        "configurations": len(EXPECTED_PROMPTS) * len(MODEL_ORDER),
        "input_rows": len(input_rows),
        "output_rows": len(output_rows),
        "token_rows": len(token_rows),
        "all_four_passages_present": True,
        "no_api_calls": True,
    }


def report_text(manifest: dict[str, Any], aggregate: list[dict[str, Any]]) -> str:
    lines = [
        "# Offline PublicHearingBR LLM cost-token collection",
        "",
        "## Classification",
        "",
        "PASS WITH WARNINGS",
        "",
        "## Executive Summary",
        "",
        "This analysis is offline and makes no API calls. It reconstructs the 4,235-example local comparison population, materializes the three published prompts, and creates canonical output proxies from stored judgments.",
        "",
        "Token counts are left unavailable when no reliable local tokenizer exists. No token count or price was invented.",
        "",
        "## Prompt Reconstruction",
        "",
        f"- Prompt files: `{manifest['prompt_dir']}`",
        "- Source: PublicHearingBR, arXiv:2410.07495v2, Appendix A, Figures 8–10.",
        "- Input hypothesis: four original passages in stored order joined with `\\n\\n`.",
        "- The hypothesis is not a claim about the original API serialization.",
        "",
        "## Tokenizer Mapping",
        "",
        "| Model | Identifier | Tokenizer | Classification | Limitation |",
        "|---|---|---|---|---|",
    ]
    seen: set[str] = set()
    for row in aggregate:
        if row["model"] in seen:
            continue
        seen.add(row["model"])
        lines.append(f"| {row['model']} | `{row['model_identifier']}` | {row['tokenizer_name']} | {row['tokenizer_classification']} | {row['notes']} |")
    lines += ["", "## Aggregated Token Counts", "", "| Model | Prompt | Input Tokens | Output Proxy Tokens | Examples |", "|---|---|---:|---:|---:|"]
    for row in aggregate:
        lines.append(f"| {row['model']} | {row['prompt_id']} | {row['total_input_tokens'] if row['total_input_tokens'] is not None else 'NA'} | {row['total_output_proxy_tokens'] if row['total_output_proxy_tokens'] is not None else 'NA'} | {row['n_examples']} |")
    lines += [
        "",
        "## Validation",
        "",
        f"- 4,235 examples: `{manifest['validation']['examples']}`",
        f"- 3 prompts: `{manifest['validation']['prompts']}`",
        f"- 4 models: `{manifest['validation']['models']}`",
        f"- 12 configurations: `{manifest['validation']['configurations']}`",
        f"- Four passages present in every input: `{manifest['validation']['all_four_passages_present']}`",
        f"- Deterministic repeat hashes: `{manifest['determinism']['pass']}`",
        f"- API calls: `{manifest['validation']['no_api_calls']}`",
        "- Population reconciliation: the paper reports 4,238 opinions; this analysis intentionally uses only the 4,235 modelable examples used by the local comparison.",
        "",
        "## Limitations",
        "",
        "- `{TEXTO}` serialization is reconstructed with an explicit `\\n\\n` separator hypothesis.",
        "- Stored judgments are converted to canonical output proxies, not recovered raw responses.",
        "- Original usage metadata, latency, request parameters, and API costs are absent.",
        "- Therefore, any later price calculation is a current-equivalent estimated cost, not a historical cost reconstruction.",
        "",
        "## Artifacts",
        "",
    ]
    for name in manifest["artifacts"]:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    prompt_dir = repo_root / "analysis/publichearing_llm_cost/prompts"
    nli_path = repo_root / "data/processed/publichearing_nli_pt_to_en_nllb/PublicHearingBR_NLI.jsonl"
    csv_path = repo_root / "results/publichearing_lora_set_transformer_mil/80eb95b5698f4a75/outputs/normalized_metadata.csv"
    output_dir = (args.output_dir or repo_root / "results/paper/llm_cost_offline").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = load_templates(prompt_dir)
    ids, stored = comparison_population_ids(nli_path)
    original = load_original_inputs(csv_path, ids)
    examples = build_examples(ids, original, stored)
    inputs = list(input_records(examples, templates))
    outputs = list(output_records(examples))
    counter = TokenCounter()
    token_rows = list(token_records(examples, templates, counter))
    validation = validate(examples, inputs, outputs, token_rows)
    aggregate = aggregate_token_rows(token_rows)

    input_digest_repeat, input_count_repeat = digest_jsonl(input_records(examples, templates))
    output_digest_repeat, output_count_repeat = digest_jsonl(output_records(examples))
    token_digest_repeat, token_count_repeat = digest_jsonl(token_rows)

    artifacts: dict[str, str] = {}
    input_path = output_dir / "reconstructed_inputs.jsonl.gz"
    output_path = output_dir / "canonical_outputs.jsonl.gz"
    token_path = output_dir / "per_example_token_counts.csv"
    aggregate_path = output_dir / "aggregated_token_counts.csv"
    input_digest, input_count = write_jsonl(input_path, inputs)
    output_digest, output_count = write_jsonl(output_path, outputs)
    token_fields = list(token_rows[0])
    token_digest, token_count = write_csv(token_path, token_fields, token_rows)
    token_rows_digest = digest_jsonl(token_rows)[0]
    aggregate_fields = list(aggregate[0])
    aggregate_digest, aggregate_count = write_csv(aggregate_path, aggregate_fields, aggregate)
    artifacts.update({
        input_path.name: sha256_file(input_path),
        output_path.name: sha256_file(output_path),
        token_path.name: sha256_file(token_path),
        aggregate_path.name: sha256_file(aggregate_path),
    })
    determinism = {
        "pass": input_digest == input_digest_repeat and output_digest == output_digest_repeat and token_rows_digest == token_digest_repeat,
        "inputs": {"first_hash": input_digest, "repeat_hash": input_digest_repeat, "first_count": input_count, "repeat_count": input_count_repeat},
        "outputs": {"first_hash": output_digest, "repeat_hash": output_digest_repeat, "first_count": output_count, "repeat_count": output_count_repeat},
        "token_rows": {"first_hash": token_rows_digest, "repeat_hash": token_digest_repeat, "csv_sha256": token_digest, "first_count": token_count, "repeat_count": token_count_repeat},
    }
    if not determinism["pass"]:
        raise ValueError("Determinism validation failed")

    manifest = {
        "status": "PASS WITH WARNINGS",
        "offline": True,
        "api_calls": 0,
        "paper_source": {
            "title": "PublicHearingBR",
            "arxiv": "2410.07495v2",
            "appendix": "A",
            "figures": [8, 9, 10],
            "url": "https://arxiv.org/pdf/2410.07495",
        },
        "prompt_dir": str(prompt_dir.relative_to(repo_root)),
        "source_files": {
            "nli_judgments": {"path": str(nli_path.relative_to(repo_root)), "sha256": sha256_file(nli_path), "opinions_in_paper": PAPER_EXAMPLES},
            "original_inputs": {"path": str(csv_path.relative_to(repo_root)), "sha256": sha256_file(csv_path)},
        },
        "population": {"comparison_examples": EXPECTED_EXAMPLES, "paper_opinions": PAPER_EXAMPLES, "difference": PAPER_EXAMPLES - EXPECTED_EXAMPLES},
        "assumptions": {"evidence_separator": "\\n\\n", "evidence_order": "stored order", "output_type": "canonical output proxy", "input_status": "RECONSTRUCTABLE WITH ASSUMPTIONS"},
        "tokenizer_mapping": {model: {"identifier": MODEL_IDENTIFIERS[model], "tokenizer": counter.info(model)[0], "classification": counter.info(model)[1], "limitation": counter.info(model)[2]} for model in MODEL_ORDER},
        "validation": validation,
        "determinism": determinism,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(report_text(manifest, aggregate), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir), "manifest": str(manifest_path), "report": str(report_path), "artifacts": artifacts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
