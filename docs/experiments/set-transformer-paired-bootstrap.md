# Gated Attention × Set Transformer paired bootstrap

## Purpose

This analysis tests the descriptive Set Transformer improvement over the
historical Gated Attention campaign on PublicHearingBR zero-shot predictions.
It is a new analysis and does not modify the frozen Attention/Mean/Max pooling
bootstrap.

The primary criterion is AUPRC. Secondary metrics are AUROC and Brier. The
delta is always `Set Transformer - Gated Attention`; therefore, a negative
Brier delta favors Set Transformer because lower Brier is better.

## Inputs and integrity

The analysis uses only existing zero-shot prediction Parquets:

- Gated Attention: `runs/ragtruth_confirmatory/4e12933c51136624/`;
- Set Transformer: `runs/ragtruth_confirmatory/28323e6cca11feb6/`.

Both campaigns use seeds 0, 1, and 2, with 4,235 examples, 206 hearings, and
501 positives per seed. Pairing is strict on `example_id`, `hearing_id`, and
`label`, and the PublicHearingBR dataset signature must match. Historical Gated
Parquets without `seed` are identified by their validated `seed_<n>` run and
manifest; Set Transformer Parquets must contain an integer `seed` column with
the corresponding value.

The historical `attention`/`gated_attention` naming difference is normalized
locally by the campaign signatures and resolved configuration. Frozen manifests
are not changed.

## Method

The comparator reuses the existing paired grouped bootstrap semantics:

- groups are `hearing_id` sampled with replacement;
- repeated draws preserve group multiplicity;
- the same sampled hearings are used for both models and all three seeds;
- each replicate computes the metric delta per seed, then averages the three
  seed deltas, preserving the campaign-level semantics of the existing pooling
  comparator;
- 10,000 valid replicas use a fixed bootstrap seed and percentile bilateral
  95% intervals.

No model, tokenizer, checkpoint, threshold, training, or GPU is used.

## Run

Validate inputs and pairing:

```bash
python scripts/run_publichearing_set_transformer_paired_grouped_bootstrap.py \
  --config configs/publichearing_set_transformer_paired_grouped_bootstrap.yaml \
  --validate-only
```

Execute the CPU-only analysis:

```bash
python scripts/run_publichearing_set_transformer_paired_grouped_bootstrap.py \
  --config configs/publichearing_set_transformer_paired_grouped_bootstrap.yaml
```

Use `--resume` to validate and reuse a completed run after an interruption or
when the result already exists:

```bash
python scripts/run_publichearing_set_transformer_paired_grouped_bootstrap.py \
  --config configs/publichearing_set_transformer_paired_grouped_bootstrap.yaml \
  --resume
```

## Outputs

The run creates:

`runs/publichearing_set_transformer_paired_bootstrap/<signature>/`

with the input hashes and paths, pairing audit, observed metrics,
`bootstrap_replicates.parquet`, `comparison_summary.json`, `report.md`,
`resolved_config.json`, and a manifest with hashes for every output artifact.

## Result scope

The campaign-level AUPRC conclusion is based on whether the paired grouped
bootstrap 95% interval for `Set Transformer - Gated Attention` crosses zero.
AUROC and Brier are secondary and are interpreted independently. This analysis
does not modify or invalidate any historical campaign or frozen bootstrap.
