#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 40
EXPECTED_EXAMPLES = 4235
EXPECTED_MODELS = ("gpt-4o", "gpt-4o-mini", "deepseek", "sabia-3.1")
EXPECTED_PROMPTS = ("prompt_1", "prompt_2", "prompt_3")
MILLION = Decimal("1000000")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000000000000001")), "f")


def ratio(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000000000000001")), "f")


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("currency") != "USD":
        raise ValueError("Pricing configuration must declare USD as its output currency")
    if set(config.get("llm_pricing", {})) != set(EXPECTED_MODELS):
        raise ValueError("Pricing configuration does not cover exactly the four LLM models")
    return config


def load_token_counts(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "model",
        "n_examples",
        "prompt_id",
        "total_input_tokens",
        "total_output_proxy_tokens",
    }
    if not required.issubset(rows[0] if rows else {}):
        raise ValueError(f"Token-count CSV is missing required fields: {sorted(required)}")
    expected_configs = {(model, prompt) for model in EXPECTED_MODELS for prompt in EXPECTED_PROMPTS}
    actual_configs = {(row["model"], row["prompt_id"]) for row in rows}
    if actual_configs != expected_configs:
        raise ValueError(f"Expected the 12 LLM configurations, found {sorted(actual_configs)}")
    if any(int(row["n_examples"]) != EXPECTED_EXAMPLES for row in rows):
        raise ValueError("Every LLM configuration must contain 4,235 examples")
    return rows


def build_cost_rows(token_rows: list[dict[str, str]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Decimal]:
    llm_pricing = config["llm_pricing"]
    set_basis = config["set_transformer"]
    set_cost = decimal(set_basis["inference_gpu_hours"]) * decimal(set_basis["total_usd_per_hour"])

    rows: list[dict[str, Any]] = []
    campaigns: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for source in token_rows:
        model = source["model"]
        pricing = llm_pricing[model]
        input_price = decimal(pricing["input_price_usd_per_million_tokens"])
        output_price = decimal(pricing["output_price_usd_per_million_tokens"])
        input_tokens = int(source["total_input_tokens"])
        output_tokens = int(source["total_output_proxy_tokens"])
        examples = int(source["n_examples"])
        cost = (Decimal(input_tokens) / MILLION) * input_price + (Decimal(output_tokens) / MILLION) * output_price
        campaigns[model] += cost
        rows.append(
            {
                "model": model,
                "pricing_model": pricing["pricing_model"],
                "pricing_basis": pricing["pricing_basis"],
                "prompt_id": source["prompt_id"],
                "n_examples": examples,
                "total_input_tokens": input_tokens,
                "total_output_proxy_tokens": output_tokens,
                "input_price_usd_per_million": format(input_price, "f"),
                "output_price_usd_per_million": format(output_price, "f"),
                "cost_usd_4235": money(cost),
                "cost_usd_per_example": money(cost / Decimal(examples)),
                "cost_usd_1m_examples": money((cost / Decimal(examples)) * MILLION),
                "cost_ratio_vs_set_transformer": ratio(cost / set_cost),
            }
        )

    set_examples = EXPECTED_EXAMPLES
    rows.append(
        {
            "model": "set-transformer",
            "pricing_model": "gcp-n1-standard-4+nvidia-t4",
            "pricing_basis": "CLOUD-EQUIVALENT",
            "prompt_id": "inference",
            "n_examples": set_examples,
            "total_input_tokens": "",
            "total_output_proxy_tokens": "",
            "input_price_usd_per_million": "",
            "output_price_usd_per_million": "",
            "cost_usd_4235": money(set_cost),
            "cost_usd_per_example": money(set_cost / Decimal(set_examples)),
            "cost_usd_1m_examples": money((set_cost / Decimal(set_examples)) * MILLION),
            "cost_ratio_vs_set_transformer": ratio(Decimal("1")),
        }
    )

    campaign_rows = [
        {
            "model": model,
            "pricing_model": llm_pricing[model]["pricing_model"],
            "prompts": "prompt_1|prompt_2|prompt_3",
            "total_cost_usd_3_prompts": money(campaigns[model]),
        }
        for model in EXPECTED_MODELS
    ]
    return rows, campaign_rows, set_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--pricing-config", type=Path, default=repo_root / "configs/publichearing_llm_cost_pricing_usd.json")
    parser.add_argument("--token-counts", type=Path, default=repo_root / "results/paper/llm_cost_offline/aggregated_token_counts.csv")
    parser.add_argument("--per-example-token-counts", type=Path, default=repo_root / "results/paper/llm_cost_offline/per_example_token_counts.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results/paper/llm_cost_offline")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    pricing_config_path = args.pricing_config.resolve()
    token_counts_path = args.token_counts.resolve()
    per_example_path = args.per_example_token_counts.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(pricing_config_path)
    token_counts_sha_before = sha256_file(token_counts_path)
    per_example_sha_before = sha256_file(per_example_path)
    token_rows = load_token_counts(token_counts_path)
    cost_rows_first, campaign_rows_first, set_cost = build_cost_rows(token_rows, config)
    cost_rows_second, campaign_rows_second, set_cost_second = build_cost_rows(token_rows, config)

    cost_fields = list(cost_rows_first[0])
    campaign_fields = list(campaign_rows_first[0])
    cost_bytes_first = csv_bytes(cost_rows_first, cost_fields)
    cost_bytes_second = csv_bytes(cost_rows_second, cost_fields)
    campaign_bytes_first = csv_bytes(campaign_rows_first, campaign_fields)
    campaign_bytes_second = csv_bytes(campaign_rows_second, campaign_fields)
    if cost_rows_first != cost_rows_second or cost_bytes_first != cost_bytes_second:
        raise ValueError("Cost calculation is not deterministic")
    if campaign_rows_first != campaign_rows_second or campaign_bytes_first != campaign_bytes_second or set_cost != set_cost_second:
        raise ValueError("Campaign calculation is not deterministic")

    cost_path = output_dir / "cost_analysis_usd.csv"
    campaign_path = output_dir / "cost_campaign_summary_usd.csv"
    manifest_path = output_dir / "cost_analysis_manifest.json"
    report_path = output_dir / "cost_analysis_report_usd.md"
    cost_path.write_bytes(cost_bytes_first)
    campaign_path.write_bytes(campaign_bytes_first)

    if sha256_file(token_counts_path) != token_counts_sha_before or sha256_file(per_example_path) != per_example_sha_before:
        raise ValueError("Token-count artifacts changed during cost calculation")

    pricing_sources = {model: config["llm_pricing"][model]["official_pricing_source"] for model in EXPECTED_MODELS}
    manifest = {
        "classification": "PASS WITH WARNINGS",
        "analysis_type": config["analysis_type"],
        "pricing_date": config["pricing_date"],
        "currency": "USD",
        "api_calls": 0,
        "pricing_sources": pricing_sources,
        "fx_conversion": config["fx_conversion"],
        "cloud_t4_pricing": {
            "provider": config["set_transformer"]["provider"],
            "gpu": config["set_transformer"]["gpu"],
            "host_configuration": config["set_transformer"]["host_configuration"],
            "region": config["set_transformer"]["region"],
            "total_usd_per_hour": config["set_transformer"]["total_usd_per_hour"],
            "pricing_date": config["set_transformer"]["pricing_date"],
            "official_gpu_pricing_source": config["set_transformer"]["official_gpu_pricing_source"],
            "official_host_pricing_source": config["set_transformer"]["official_host_pricing_source"],
        },
        "set_transformer_basis": {
            "examples": EXPECTED_EXAMPLES,
            "inference_seconds": config["set_transformer"]["inference_seconds"],
            "inference_gpu_hours": config["set_transformer"]["inference_gpu_hours"],
            "configuration_source": config["set_transformer"]["configuration_source"],
            "inference_measurement_sources": config["set_transformer"]["inference_measurement_sources"],
            "cost_usd_4235": money(set_cost),
            "development_only_training_gpu_hours_three_seeds": config["set_transformer"]["development_only_training_gpu_hours_three_seeds"],
            "training_included_in_inference_cost": False,
        },
        "source_artifacts": {
            "pricing_config": {"path": str(pricing_config_path.relative_to(repo_root)), "sha256": sha256_file(pricing_config_path)},
            "aggregated_token_counts": {"path": str(token_counts_path.relative_to(repo_root)), "sha256": token_counts_sha_before},
            "per_example_token_counts": {"path": str(per_example_path.relative_to(repo_root)), "sha256": per_example_sha_before},
        },
        "assumptions": [
            "Input and output token counts are consumed exactly as present in aggregated_token_counts.csv.",
            "Each LLM x prompt is compared individually against one Set Transformer inference run.",
            "The three prompts are summed only in cost_campaign_summary_usd.csv as a full prompting campaign.",
            "OpenAI uses standard uncached text-token rates.",
            "DeepSeek uses peak cache-miss Flash rates and no cache discount.",
            "Sabiá provider-local prices are converted to USD using the recorded official PTAX midpoint; monetary outputs contain USD only.",
            "The Set Transformer hourly basis is GPU plus host compute; disk, network, tax, and management charges are excluded.",
            "The Set Transformer runtime and GPU-hours are approximate measurements supplied for the primary condition.",
        ],
        "validation": {
            "llm_configurations": len(token_rows),
            "examples_per_llm_configuration": sorted({int(row["n_examples"]) for row in token_rows}),
            "all_monetary_outputs_usd": True,
            "zero_api_calls": True,
            "all_prices_have_official_sources": True,
            "cache_discounts_applied": False,
            "deterministic_calculation": True,
            "aggregated_token_counts_hash_preserved": sha256_file(token_counts_path) == token_counts_sha_before,
            "per_example_token_counts_hash_preserved": sha256_file(per_example_path) == per_example_sha_before,
        },
        "artifacts": {
            cost_path.name: sha256_file(cost_path),
            campaign_path.name: sha256_file(campaign_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PublicHearingBR current-equivalent cost analysis (USD)",
        "",
        "## Classification",
        "",
        "PASS WITH WARNINGS",
        "",
        "## Pricing Basis",
        "",
        "| Model | Pricing model | Basis | Input USD/M | Output USD/M | Source |",
        "|---|---|---|---:|---:|---|",
    ]
    for model in EXPECTED_MODELS:
        pricing = config["llm_pricing"][model]
        lines.append(
            f"| {model} | `{pricing['pricing_model']}` | {pricing['pricing_basis']} | {pricing['input_price_usd_per_million_tokens']} | {pricing['output_price_usd_per_million_tokens']} | [{pricing['official_pricing_source']}]({pricing['official_pricing_source']}) |"
        )
    lines += [
        "",
        "## Set Transformer Basis",
        "",
        "| GPU | Runtime | GPU-hours | USD/hour | Cost for 4,235 |",
        "|---|---:|---:|---:|---:|",
        f"| {config['set_transformer']['provider']} {config['set_transformer']['gpu']} + {config['set_transformer']['host_configuration']} | {config['set_transformer']['inference_seconds']} s | {config['set_transformer']['inference_gpu_hours']} | {config['set_transformer']['total_usd_per_hour']} | {money(set_cost)} |",
        "",
        f"Measurement configuration: `{config['set_transformer']['configuration_source']}`; inference logs: " + ", ".join(f"`{path}`" for path in config["set_transformer"]["inference_measurement_sources"]),
        "",
        f"Development-only reference: three seeds = {config['set_transformer']['development_only_training_gpu_hours_three_seeds']} T4 GPU-hours; not included in inference cost and not treated as an ensemble.",
        "",
        "## Estimated Cost",
        "",
        "| Model | Prompt | Cost/4,235 USD | Cost/example USD | Cost/1M USD | Ratio vs ours |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in cost_rows_first:
        lines.append(f"| {row['model']} | {row['prompt_id']} | {row['cost_usd_4235']} | {row['cost_usd_per_example']} | {row['cost_usd_1m_examples']} | {row['cost_ratio_vs_set_transformer']} |")
    lines += [
        "",
        "## Full Prompting Campaign",
        "",
        "| Model | Cost of all 3 prompts USD |",
        "|---|---:|",
    ]
    for row in campaign_rows_first:
        lines.append(f"| {row['model']} | {row['total_cost_usd_3_prompts']} |")
    lines += [
        "",
        "## Limitations",
        "",
        "- These are current-equivalent estimated costs, not the authors' historical invoices.",
        "- Inputs were reconstructed offline; the original provider request logs were not available.",
        "- LLM outputs are canonical output proxies, not original completion usage metadata.",
        "- DeepSeek and Sabiá use successor pricing; DeepSeek's current V4.1-Flash billing target differs from the earlier V4 tokenizer artifact used by the preserved token counts.",
        "- The internal provider chat protocol may add a small amount of overhead not represented by content-only token counts.",
        "- The cloud T4 basis excludes disk, network, tax, and management charges.",
        "- No API calls were made and no prompt, output, or token-count artifact was modified.",
        "- No LaTeX text was generated.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": manifest["classification"],
        "set_transformer_cost_usd_4235": money(set_cost),
        "cost_analysis": str(cost_path),
        "campaign_summary": str(campaign_path),
        "manifest": str(manifest_path),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
