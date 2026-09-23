#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_EXAMPLES = 4235
EXPECTED_PROMPTS = ("prompt_1", "prompt_2", "prompt_3")
EXPECTED_MODELS = ("gpt-4o", "gpt-4o-mini", "deepseek", "sabia-3.1")
MODEL_IDENTIFIERS = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "deepseek": "deepseek-chat",
    "sabia-3.1": "sabia-3.1-2025-05-08",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Tokenizers:
    def __init__(self, o200k_path: Path, deepseek_dir: Path) -> None:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        from transformers import AutoTokenizer
        from maritalk import count_tokens

        self.count_sabia = count_tokens
        self.o200k_path = o200k_path.resolve()
        self.deepseek_dir = deepseek_dir.resolve()
        expected_o200k_sha256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
        if sha256_file(self.o200k_path) != expected_o200k_sha256:
            raise ValueError("o200k_base.tiktoken hash does not match the official tiktoken artifact")
        mergeable_ranks = load_tiktoken_bpe(str(self.o200k_path), expected_hash=expected_o200k_sha256)
        pat_str = "|".join(
            [
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""\p{N}{1,3}""",
                r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                r"""\s*[\r\n]+""",
                r"""\s+(?!\S)""",
                r"""\s+""",
            ]
        )
        self.openai_encoder = tiktoken.Encoding(
            name="o200k_base",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
        )
        self.deepseek_tokenizer = AutoTokenizer.from_pretrained(
            self.deepseek_dir,
            local_files_only=True,
            trust_remote_code=True,
        )

    def count_input(self, model: str, record: dict[str, Any]) -> int:
        system = record["system_message"]
        user = record["user_message"]
        if model in {"gpt-4o", "gpt-4o-mini"}:
            return len(self.openai_encoder.encode(system + "\n" + user, disallowed_special=()))
        if model == "deepseek":
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return len(self.deepseek_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if model == "sabia-3.1":
            return int(self.count_sabia(system + "\n" + user, model="sabia-4"))
        raise ValueError(f"Unknown model: {model}")

    def count_output(self, model: str, text: str) -> int:
        if model in {"gpt-4o", "gpt-4o-mini"}:
            return len(self.openai_encoder.encode(text, disallowed_special=()))
        if model == "deepseek":
            return len(self.deepseek_tokenizer.encode(text, add_special_tokens=False))
        if model == "sabia-3.1":
            return int(self.count_sabia(text, model="sabia-4"))
        raise ValueError(f"Unknown model: {model}")

    def mapping(self) -> dict[str, dict[str, str]]:
        return {
            "gpt-4o": {
                "original_model_identifier": MODEL_IDENTIFIERS["gpt-4o"],
                "tokenizer_target_model": "gpt-4o-2024-08-06",
                "tokenizer_name": "o200k_base",
                "tokenizer_source": "OpenAI tiktoken official o200k_base artifact",
                "tokenizer_classification": "EXACT",
                "rationale": "The pinned GPT-4o snapshot is documented with the o200k tokenizer family; the official static artifact is hash-verified. Counts are content-only and exclude API chat-wrapper overhead.",
            },
            "gpt-4o-mini": {
                "original_model_identifier": MODEL_IDENTIFIERS["gpt-4o-mini"],
                "tokenizer_target_model": "gpt-4o-mini-2024-07-18",
                "tokenizer_name": "o200k_base",
                "tokenizer_source": "OpenAI tiktoken official o200k_base artifact",
                "tokenizer_classification": "EXACT",
                "rationale": "The pinned GPT-4o mini snapshot uses the same o200k tokenizer family; the official static artifact is hash-verified. Counts are content-only and exclude API chat-wrapper overhead.",
            },
            "deepseek": {
                "original_model_identifier": MODEL_IDENTIFIERS["deepseek"],
                "tokenizer_target_model": "deepseek-v4-flash",
                "tokenizer_name": "DeepSeek V4 official tokenizer.json + chat template",
                "tokenizer_source": "DeepSeek official deepseek_v4_tokenizer.zip",
                "tokenizer_classification": "OFFICIAL SUCCESSOR",
                "rationale": "The historical deepseek-chat alias was upgraded and then discontinued; the current-equivalent target is the V4 Flash non-thinking path. Input counts use the official V4 chat template.",
            },
            "sabia-3.1": {
                "original_model_identifier": MODEL_IDENTIFIERS["sabia-3.1"],
                "tokenizer_target_model": "sabia-4",
                "tokenizer_name": "Maritaca maritalk bundled Sabiá-4 encoder",
                "tokenizer_source": "maritalk package tokenizer_data_v4 / Maritaca count_tokens",
                "tokenizer_classification": "APPROXIMATE",
                "rationale": "Sabiá-4 is the current API family target and has an official local count_tokens encoder. The current documentation does not explicitly map the historical Sabiá-3.1 identifier to a replacement, so this successor mapping is an explicit approximation.",
            },
        }


def build_token_rows(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]], tokenizers: Tokenizers, mapping: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    input_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in inputs:
        key = (record["example_id"], record["prompt_id"])
        if key in input_lookup:
            raise ValueError(f"Duplicate input key: {key}")
        input_lookup[key] = record
    if len(input_lookup) != EXPECTED_EXAMPLES * len(EXPECTED_PROMPTS):
        raise ValueError(f"Expected {EXPECTED_EXAMPLES * len(EXPECTED_PROMPTS)} input records, found {len(input_lookup)}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for output in outputs:
        model = output["model"]
        prompt_id = output["prompt_id"]
        key = (output["example_id"], model, prompt_id)
        if key in seen:
            raise ValueError(f"Duplicate output key: {key}")
        seen.add(key)
        input_record = input_lookup[(output["example_id"], prompt_id)]
        meta = mapping[model]
        rows.append(
            {
                "example_id": output["example_id"],
                "model": model,
                "model_identifier": MODEL_IDENTIFIERS[model],
                "prompt_id": prompt_id,
                "input_tokens": tokenizers.count_input(model, input_record),
                "output_proxy_tokens": tokenizers.count_output(model, output["canonical_output"]),
                "tokenizer_target_model": meta["tokenizer_target_model"],
                "tokenizer_name": meta["tokenizer_name"],
                "tokenizer_classification": meta["tokenizer_classification"],
                "input_reconstruction_status": "RECONSTRUCTED INPUT; MODEL-SPECIFIC TOKENIZATION",
                "output_reconstruction_status": "CANONICAL OUTPUT PROXY",
                "notes": meta["rationale"],
            }
        )
    expected = EXPECTED_EXAMPLES * len(EXPECTED_PROMPTS) * len(EXPECTED_MODELS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} token rows, found {len(rows)}")
    return rows


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["prompt_id"])].append(row)
    result: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        for prompt_id in EXPECTED_PROMPTS:
            group = groups[(model, prompt_id)]
            input_values = [row["input_tokens"] for row in group]
            output_values = [row["output_proxy_tokens"] for row in group]
            first = group[0]
            result.append(
                {
                    "model": model,
                    "model_identifier": first["model_identifier"],
                    "tokenizer_target_model": first["tokenizer_target_model"],
                    "prompt_id": prompt_id,
                    "n_examples": len(group),
                    "total_input_tokens": sum(input_values),
                    "mean_input_tokens": statistics.mean(input_values),
                    "median_input_tokens": statistics.median(input_values),
                    "total_output_proxy_tokens": sum(output_values),
                    "mean_output_proxy_tokens": statistics.mean(output_values),
                    "median_output_proxy_tokens": statistics.median(output_values),
                    "tokenizer_name": first["tokenizer_name"],
                    "tokenizer_classification": first["tokenizer_classification"],
                    "input_reconstruction_status": first["input_reconstruction_status"],
                    "output_reconstruction_status": first["output_reconstruction_status"],
                    "notes": first["notes"],
                }
            )
    return result


def write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return sha256_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--o200k-path", type=Path, default=Path("/tmp/publichearing_tokenizers/o200k_base.tiktoken"))
    parser.add_argument("--deepseek-tokenizer-dir", type=Path, default=Path("/tmp/publichearing_tokenizers/deepseek_v4/deepseek_v4_tokenizer"))
    parser.add_argument("--deepseek-zip-path", type=Path, default=Path("/tmp/publichearing_tokenizers/deepseek_v4_tokenizer.zip"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir or repo_root / "results/paper/llm_cost_offline").resolve()
    input_path = output_dir / "reconstructed_inputs.jsonl.gz"
    output_path = output_dir / "canonical_outputs.jsonl.gz"
    per_example_path = output_dir / "per_example_token_counts.csv"
    aggregate_path = output_dir / "aggregated_token_counts.csv"
    mapping_path = output_dir / "tokenizer_mapping.json"
    manifest_path = output_dir / "tokenization_manifest.json"
    report_path = output_dir / "tokenization_report.md"

    input_sha_before = sha256_file(input_path)
    output_sha_before = sha256_file(output_path)
    inputs = load_jsonl_gz(input_path)
    outputs = load_jsonl_gz(output_path)
    if len(inputs) != EXPECTED_EXAMPLES * len(EXPECTED_PROMPTS):
        raise ValueError(f"Expected 12,705 reconstructed inputs, found {len(inputs)}")
    if len(outputs) != EXPECTED_EXAMPLES * len(EXPECTED_PROMPTS) * len(EXPECTED_MODELS):
        raise ValueError(f"Expected 50,820 canonical outputs, found {len(outputs)}")
    if len({row["example_id"] for row in inputs}) != EXPECTED_EXAMPLES:
        raise ValueError("Input example IDs do not cover exactly 4,235 examples")
    if any(len(row.get("evidence_passages", [])) != 4 for row in inputs):
        raise ValueError("An input does not contain exactly four evidence passages")

    tokenizers = Tokenizers(args.o200k_path, args.deepseek_tokenizer_dir)
    mapping = tokenizers.mapping()
    first_rows = build_token_rows(inputs, outputs, tokenizers, mapping)
    second_rows = build_token_rows(inputs, outputs, tokenizers, mapping)
    fields = list(first_rows[0])
    if first_rows != second_rows:
        raise ValueError("Token counting is not deterministic")
    first_per_example = csv_bytes(first_rows, fields)
    second_per_example = csv_bytes(second_rows, fields)
    if first_per_example != second_per_example:
        raise ValueError("Per-example CSV serialization is not deterministic")
    first_aggregate = aggregate(first_rows)
    second_aggregate = aggregate(second_rows)
    aggregate_fields = list(first_aggregate[0])
    first_aggregate_bytes = csv_bytes(first_aggregate, aggregate_fields)
    second_aggregate_bytes = csv_bytes(second_aggregate, aggregate_fields)
    if first_aggregate_bytes != second_aggregate_bytes:
        raise ValueError("Aggregate CSV serialization is not deterministic")

    per_example_sha = write_bytes(per_example_path, first_per_example)
    aggregate_sha = write_bytes(aggregate_path, first_aggregate_bytes)
    mapping_sha = write_bytes(mapping_path, (json.dumps(mapping, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if sha256_file(input_path) != input_sha_before or sha256_file(output_path) != output_sha_before:
        raise ValueError("Existing prompt/output reconstruction artifacts changed")

    tokenizer_files = {
        "o200k_base": {"path": str(args.o200k_path.resolve()), "sha256": sha256_file(args.o200k_path)},
        "deepseek_v4_zip": {"path": str(args.deepseek_zip_path.resolve()), "sha256": None},
    }
    zip_candidate = args.deepseek_zip_path
    if zip_candidate.is_file():
        tokenizer_files["deepseek_v4_zip"]["sha256"] = sha256_file(zip_candidate)

    manifest = {
        "classification": "PASS WITH WARNINGS",
        "offline": True,
        "api_calls": 0,
        "source_artifacts": {
            "reconstructed_inputs": {"path": str(input_path.relative_to(repo_root)), "sha256": input_sha_before},
            "canonical_outputs": {"path": str(output_path.relative_to(repo_root)), "sha256": output_sha_before},
        },
        "population": {"examples": EXPECTED_EXAMPLES, "prompts": len(EXPECTED_PROMPTS), "models": len(EXPECTED_MODELS), "configurations": 12},
        "counting_method": {
            "openai_and_sabia": "system_message + '\\n' + user_message; no API chat-wrapper overhead",
            "deepseek": "official V4 chat_template with system/user messages and add_generation_prompt=True",
            "output": "canonical_output text only; no assistant wrapper or historical usage metadata",
        },
        "tokenizer_mapping": mapping,
        "tokenizer_files": tokenizer_files,
        "dependencies": {
            "tiktoken": __import__("tiktoken").__version__,
            "maritalk": __import__("importlib.metadata", fromlist=["version"]).version("maritalk"),
            "transformers": __import__("transformers").__version__,
        },
        "validation": {
            "input_rows": len(inputs),
            "output_rows": len(outputs),
            "token_rows": len(first_rows),
            "all_four_passages_present": True,
            "deterministic_token_counts": True,
            "zero_api_calls": True,
        },
        "artifacts": {
            per_example_path.name: per_example_sha,
            aggregate_path.name: aggregate_sha,
            mapping_path.name: mapping_sha,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PublicHearingBR offline tokenization",
        "",
        "## Classification",
        "",
        "PASS WITH WARNINGS",
        "",
        "## Tokenizer Mapping",
        "",
        "| Model | Original identifier | Tokenizer target | Tokenizer | Classification | Rationale |",
        "|---|---|---|---|---|---|",
    ]
    for model in EXPECTED_MODELS:
        item = mapping[model]
        lines.append(f"| {model} | `{item['original_model_identifier']}` | `{item['tokenizer_target_model']}` | {item['tokenizer_name']} | {item['tokenizer_classification']} | {item['rationale']} |")
    lines += ["", "## Aggregated Token Counts", "", "| Model | Prompt | Examples | Input tokens | Output proxy tokens |", "|---|---|---:|---:|---:|"]
    for row in first_aggregate:
        lines.append(f"| {row['model']} | {row['prompt_id']} | {row['n_examples']} | {row['total_input_tokens']} | {row['total_output_proxy_tokens']} |")
    lines += [
        "",
        "The CSV also contains mean and median counts per configuration.",
        "",
        "## Validation",
        "",
        "- 4,235 examples: PASS",
        "- 3 prompts: PASS",
        "- 4 models: PASS",
        "- 12 configurations: PASS",
        "- deterministic token counts: PASS",
        "- zero API calls: PASS",
        "- reconstructed input/output gzip hashes unchanged: PASS",
        "",
        "## Warnings",
        "",
        "- GPT counts use the exact o200k tokenizer artifact, but count content-only System/User text and exclude provider chat-wrapper overhead.",
        "- DeepSeek counts target the current-equivalent V4 Flash tokenizer and therefore are not counts for the historical DeepSeek-V3 tokenizer.",
        "- Sabiá counts target Sabiá 4. The current documentation does not explicitly state a Sabiá-3.1 replacement, so this mapping is APPROXIMATE and must remain paired with Sabiá 4 pricing later.",
        "- Output counts are canonical proxy text counts, not original API completion usage.",
        "- No monetary cost was calculated.",
        "",
        "## Artifacts",
        "",
        f"- `{per_example_path.name}`",
        f"- `{aggregate_path.name}`",
        f"- `{mapping_path.name}`",
        f"- `{manifest_path.name}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": manifest["classification"], "report": str(report_path), "manifest": str(manifest_path), "per_example_sha256": per_example_sha, "aggregate_sha256": aggregate_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
