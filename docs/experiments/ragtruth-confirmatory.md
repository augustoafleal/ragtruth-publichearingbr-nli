# RAGTruth confirmatory experiment

## Purpose

This experiment trains three LoRA seeds on the frozen deduplicated RAGTruth
training view. It evaluates the RAGTruth test split and produces frozen
zero-shot predictions for PublicHearingBR.

## Required preparation

Build the RAGTruth training view before this experiment. See
[RAGTruth data](../preparation/ragtruth-data.md).

The confirmatory configuration records the expected Parquet dataset and grouped
split. Verify both values in the resolved configuration before training.

## Run

Script: `scripts/run_ragtruth_confirmatory.py`

Validate inputs before training:

```bash
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --validate-only
```

Training phase:

```bash
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --phase train
```

Evaluation phase:

```bash
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --phase evaluate
```

Aggregation phase:

```bash
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --phase aggregate
```

## Outputs

The campaign creates `runs/ragtruth_confirmatory/<signature>/` with one
directory per seed, test predictions, PublicHearingBR zero-shot predictions,
thresholds, split assignments, aggregate metrics, and manifests.

## Result scope

The campaign uses seeds 0, 1, and 2. It does not select a best seed and does
not create an ensemble.
