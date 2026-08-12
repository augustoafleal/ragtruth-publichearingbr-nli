# Continuous paired grouped bootstrap

## Purpose

This analysis compares frozen continuous scores from the off the shelf baseline
and the three RAGTruth LoRA seeds. It estimates uncertainty for AUPRC, AUROC,
and Brier improvement.

## Required preparation

Complete the [RAGTruth confirmatory experiment](ragtruth-confirmatory.md) and
the [off the shelf NLI baseline](off-the-shelf-nli.md). This analysis reads
frozen prediction files and does not need a model, tokenizer, CUDA, inference,
or training.

## Run

Validate input pairing first:

```bash
python scripts/run_publichearing_paired_grouped_bootstrap.py \
  --config configs/publichearing_paired_grouped_bootstrap.yaml \
  --validate-only
```

Run the bootstrap:

```bash
python scripts/run_publichearing_paired_grouped_bootstrap.py \
  --config configs/publichearing_paired_grouped_bootstrap.yaml
```

## Outputs

The run creates `runs/publichearing_paired_grouped_bootstrap/<signature>/`.
It stores bootstrap replicates, observed metrics, per seed results, campaign
summary, integrity audit, configuration, manifest, and report.

## Result scope

The sampling unit is `hearing_id`. The same grouped samples are used for the
baseline and all LoRA seeds. The two existing signatures are duplicate
materializations of one scientific calculation.
