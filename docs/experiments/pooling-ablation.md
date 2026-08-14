# Pooling ablation: Gated Attention, Mean, Max, and Set Transformer

## Purpose

This controlled ablation compares Gated Attention, Mean, Max, and Set
Transformer aggregation in the RAGTruth confirmatory transfer pipeline. The
only intended scientific difference is pooling over the projected evidence
representations; the encoder, projection, classifier, dataset, retrieval Top-4,
split, seeds, optimization, and checkpoint selection remain fixed.

## Required preparation

Build the RAGTruth training view before this experiment. See
[RAGTruth data](../preparation/ragtruth-data.md). The Mean and Max
configurations use the same signed Parquet dataset and grouped split as the
RAGTruth confirmatory campaign.

## Run

The Gated Attention campaign is the legacy confirmatory run. To run each
ablation variant, validate inputs and complete each phase separately.

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

## Set Transformer

The Set Transformer configuration uses one Set Attention Block (SAB) and
Pooling by Multihead Attention (PMA) with one learned seed. It operates on the
same `[B, 4, 128]` projected evidence representations as the other methods,
uses four heads, a 128-dimensional internal FFN, residual connections,
LayerNorm, and no positional encoding. The SAB and PMA both consume
`evidence_mask`; masked slots cannot act as keys/values or contribute to the
final representation. Consequently, the aggregator is invariant to a joint
permutation of chunks and their masks. ISAB/inducing points are not used
because each bag has exactly four slots.

`configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml` fixes
`architecture: set_transformer`, `pooling_type: set_transformer`, one SAB,
four heads, one PMA seed, and a 128-dimensional FFN. These aggregator settings
are included in the resolved configuration, manifest, and scientific
fingerprint, so Set Transformer checkpoints are distinct from Gated Attention
checkpoints.

PMA emits head-averaged `[B, 4]` weights for the existing inference/audit
contract. They are diagnostics only: unlike Gated Attention weights, their
keys have already been contextualized by the SAB and are not semantically
equivalent attribution scores.

```bash
# Validate the frozen inputs and configuration
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml \
  --validate-only

# Train seeds 0, 1, and 2; add --resume only after an interruption
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml \
  --phase train

# Evaluate the selected checkpoints, including PublicHearingBR zero-shot
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml \
  --phase evaluate

# Aggregate the three seed-level outputs
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml \
  --phase aggregate
```

The existing Attention/Mean/Max paired bootstrap is frozen and intentionally
does not include this campaign. A future Gated-versus-Set comparator must also
resolve its historical naming mismatch: the current canonical Gated pooling
name is `attention`, while that frozen bootstrap identifies its reference
campaign as `gated_attention`.

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
