# Offline LLM tokenization and current-equivalent cost analysis

## Purpose

This analysis estimates the current-equivalent inference cost of the four
PublicHearingBR LLM comparators and one Set Transformer inference run on a
cloud-equivalent Tesla T4. It is fully offline: it does not call provider APIs,
recover historical invoices, or modify the reconstructed prompts, canonical
outputs, or token-count artifacts.

The LLM comparison covers GPT-4o, GPT-4o mini, DeepSeek, and Sabiá across the
three reconstructed prompt configurations. The Set Transformer comparison uses
one trained model evaluated on the same 4,235 examples.

## Required preparation

The reconstruction stage requires the PublicHearingBR NLI source and the
normalized Set Transformer input artifact referenced by
`scripts/reconstruct_publichearing_llm_cost.py`. The tokenizer stage requires
the local tokenizer artifacts recorded in
`results/paper/llm_cost_offline/tokenization_manifest.json`.

The cost stage requires:

- `results/paper/llm_cost_offline/aggregated_token_counts.csv`;
- `results/paper/llm_cost_offline/per_example_token_counts.csv`;
- `configs/publichearing_llm_cost_pricing_usd.json`;
- the Set Transformer inference measurement and run manifest recorded in the
  pricing configuration.

The pricing configuration contains only USD output prices. The Sabiá exchange
rate is retained as provenance in the configuration and manifest.

## Run

Reconstruct the frozen local inputs and canonical output proxies:

```bash
python scripts/reconstruct_publichearing_llm_cost.py
```

Count tokens using the mapped local tokenizer artifacts:

```bash
python scripts/count_publichearing_llm_tokens.py
```

Calculate the USD estimates from the existing token counts:

```bash
python scripts/calculate_publichearing_cost_analysis_usd.py
```

The final command reads the token-count CSVs, verifies their hashes before and
after calculation, performs the calculation twice for determinism, and writes
only the cost-analysis outputs and report.

## Pricing and measurement basis

GPT-4o and GPT-4o mini use their pinned current model identifiers. DeepSeek uses
the current `deepseek-flash` successor with the standard cache-miss tariff;
cache discounts are not applied. Sabiá uses `sabia-4` as the current successor
pricing target, with the documented USD conversion provenance.

The Set Transformer uses one inference run on a Tesla T4, approximately 593
seconds or 0.164657 GPU-hours. The cloud-equivalent basis is Google Cloud
`n1-standard-4` plus one NVIDIA T4 in `us-central1`, with GPU and host charges
combined. The three-seed development total of 6.948743 T4 GPU-hours is retained
as provenance but is excluded from inference cost and is not treated as an
ensemble.

## Outputs

The analysis writes `results/paper/llm_cost_offline/` with:

- `per_example_token_counts.csv` and `aggregated_token_counts.csv`;
- `tokenizer_mapping.json`, `tokenization_manifest.json`, and
  `tokenization_report.md`;
- `cost_analysis_usd.csv` with one row per LLM × prompt and one Set Transformer
  inference row;
- `cost_campaign_summary_usd.csv` with the three-prompt campaign total for each
  LLM;
- `cost_analysis_manifest.json` with pricing sources, hashes, assumptions, and
  validation fields;
- `cost_analysis_report_usd.md` with the human-readable USD report.

The versioned pricing input is
`configs/publichearing_llm_cost_pricing_usd.json`.

## Result scope

The primary comparison is one LLM prompt configuration against one Set
Transformer inference run. The three prompts are summed only in the campaign
summary; they are not the primary per-configuration comparison.

The results are current-equivalent estimated costs, not historical author
invoices. Inputs were reconstructed from the available local artifacts, and
LLM responses are canonical output proxies rather than raw provider responses.
DeepSeek and Sabiá use successor pricing/tokenization targets. Provider chat
protocol overhead is not included in the content-level token counts.

The current validated run contains 12 LLM configurations, 4,235 examples per
configuration, zero provider API calls, USD-only monetary outputs, and
deterministic cost calculations.
