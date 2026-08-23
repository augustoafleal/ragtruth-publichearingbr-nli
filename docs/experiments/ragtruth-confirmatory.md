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
evaluation protocol, while selecting the canonical translated Portuguese
Parquet. It has the same 34,604 rows and source IDs as the English view; only
the claim and valid evidence text are translated. Grouped assignments are
compared with the English reference during translation.

The compatibility wrapper only dispatches the confirmatory flow; it does not
filter or rebuild data:

```bash
python scripts/run_ragtruth_pt_nllb_experiment.py \
  --config configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml \
  --dry-run

python scripts/run_ragtruth_pt_nllb_experiment.py \
  --config configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml
```

The validation-only command does not download model weights, initialize CUDA,
or start training. During training, RAGTruth validation is used to select
checkpoints and thresholds. RAGTruth test and PublicHearingBR zero-shot
evaluation run only afterward. The PT zero-shot configuration expects the
dataset at `data/PublicHearingBR_NLI.jsonl`. Make sure it is available before
the evaluation phase.

## Portuguese MADLAD condition

The MADLAD condition uses the same confirmatory architecture, optimizer,
seeds, epoch budget and selection protocol as NLLB. Its translated Parquet is
`data/processed/ragtruth_confirmatory_pt_madlad/dataset.parquet`, and its
configuration is `configs/ragtruth_pt_madlad_lora_attention_mil_confirmatory.yaml`.
The condition uses `longest_first` token truncation because translated claims
can exceed the 512-token model limit. The dedicated PublicHearingBR
configuration is `configs/ragtruth_pt_madlad_to_publichearing_zero_shot.yaml`.

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
