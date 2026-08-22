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

## Portuguese NLLB condition

`configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml` keeps
the same model, LoRA settings, seeds, optimizer, epochs, validation metric and
evaluation protocol, while selecting only the filtered Portuguese JSONL
directory. The filter manifest is frozen by its input hashes, aggregate dataset
hash and split signature. It currently contains 45,238 train, 8,143 validation
and 9,121 test examples.

The one-shot wrapper validates/reuses the filtered dataset and then starts
training:

```bash
python scripts/run_ragtruth_pt_nllb_experiment.py \
  --config configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml \
  --dry-run

python scripts/run_ragtruth_pt_nllb_experiment.py \
  --config configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml
```

Use `--force-filter` only when intentionally rebuilding the filtered output.
The validation-only command does not download model weights, initialize CUDA,
or execute training. The training phase uses RAGTruth validation only for
checkpoint/threshold selection; RAGTruth test and PublicHearingBR zero-shot
evaluation remain downstream phases. The PT companion zero-shot config uses
the relative path `data/PublicHearingBR_NLI.jsonl`; stage that dataset before
the evaluation phase.

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

## Controlled pooling ablation

The legacy Gated Attention campaign is also compared with Mean, Max, and Set
Transformer pooling under the same frozen RAGTruth-to-PublicHearingBR protocol. See
[Pooling ablation](pooling-ablation.md) for the variant configurations,
signatures, results, and scope of the comparison.
