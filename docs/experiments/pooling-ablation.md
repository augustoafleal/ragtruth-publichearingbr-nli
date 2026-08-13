# Pooling ablation: Gated Attention, Mean, and Max

## Purpose

This controlled ablation compares Gated Attention, Mean, and Max aggregation in
the RAGTruth confirmatory transfer pipeline. The only intended scientific
difference is pooling over the projected evidence representations; the encoder,
projection, classifier, dataset, retrieval Top-4, split, seeds, optimization,
and checkpoint selection remain fixed.

## Required preparation

Build the RAGTruth training view before this experiment. See
[RAGTruth data](../preparation/ragtruth-data.md). The Mean and Max
configurations use the same signed Parquet dataset and grouped split as the
RAGTruth confirmatory campaign.

## Run

The Gated Attention campaign is the legacy confirmatory run. To reproduce the
two ablation variants, validate inputs and complete each phase separately.

```bash
# Mean
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_mean_mil_confirmatory.yaml \
  --validate-only
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_mean_mil_confirmatory.yaml \
  --phase train
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_mean_mil_confirmatory.yaml \
  --phase evaluate
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_mean_mil_confirmatory.yaml \
  --phase aggregate
```

```bash
# Max
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_max_mil_confirmatory.yaml \
  --validate-only
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_max_mil_confirmatory.yaml \
  --phase train
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_max_mil_confirmatory.yaml \
  --phase evaluate
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_max_mil_confirmatory.yaml \
  --phase aggregate
```

Use `--resume` with the training phase after an interruption.

## Outputs

Each campaign writes `runs/ragtruth_confirmatory/<signature>/` with one
directory per seed, RAGTruth test predictions, PublicHearingBR zero-shot
predictions, thresholds, aggregate metrics, configuration, and manifests.

## Result scope

The ablation uses seeds 0, 1, and 2. It does not select a best seed or create
an ensemble. Thresholds are selected separately per pooling and seed on
RAGTruth validation only, then frozen before RAGTruth test and
PublicHearingBR zero-shot evaluation.

The existing paired grouped bootstraps compare Gated Attention with the
off-the-shelf NLI baseline; they do not estimate uncertainty between pooling
variants. Pooling comparisons are therefore descriptive.
