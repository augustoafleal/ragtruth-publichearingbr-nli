# Thresholded paired grouped bootstrap

## Purpose

This analysis estimates uncertainty for F1, recall, precision, MCC, balanced
accuracy, FPR, specificity, and accuracy after applying frozen thresholds. It
compares the baseline and each RAGTruth LoRA seed with paired hearing samples.

## Required preparation

Complete [threshold transfer](threshold-transfer.md). That run provides the
frozen `best_f1` and `fpr10` thresholds. The bootstrap reads frozen predictions
and does not need a model, tokenizer, CUDA, inference, or training.

## Run

Validate input hashes, pairing, and thresholds first:

```bash
python scripts/run_publichearing_thresholded_paired_bootstrap.py \
  --config configs/publichearing_thresholded_paired_bootstrap.yaml \
  --validate-only
```

Run the analysis:

```bash
python scripts/run_publichearing_thresholded_paired_bootstrap.py \
  --config configs/publichearing_thresholded_paired_bootstrap.yaml
```

## Outputs

The run creates `runs/publichearing_thresholded_paired_bootstrap/<signature>/`.
It stores observed metrics, observed effects, threshold audit, bootstrap
replicates, per seed results, campaign summary, integrity audit, configuration,
manifest, and report.

## Result scope

Thresholds remain constant in every bootstrap replicate. A lower FPR alone is
not treated as global superiority because the baseline is very conservative.
