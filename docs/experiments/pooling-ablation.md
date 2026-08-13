# Pooling ablation: Gated Attention, Mean, and Max

## Purpose

This controlled ablation compares the aggregation applied to the same projected
evidence representations in the RAGTruth confirmatory transfer pipeline:

```text
claim + evidence_i -> mDeBERTa + LoRA -> CLS -> Linear(H, 128) + GELU + Dropout
                   -> pooling -> Dropout + Linear(128, 1) -> hallucination score
```

The only intended scientific difference is pooling over `projected [B,N,128]`:

- **Gated Attention** uses the existing trainable gated MIL scorer.
- **Mean** averages valid evidence representations.
- **Max** takes the feature-wise maximum over valid evidence representations.

The projection, LoRA adapters, classifier, dataset, retrieval Top-4, grouped
split, seeds, optimization, and checkpoint selection remain fixed. Gated
Attention alone has additional pooling parameters; Mean and Max are
non-parametric.

## Required preparation

Complete the [RAGTruth confirmatory experiment](ragtruth-confirmatory.md) and
build the signed RAGTruth training view described in [RAGTruth data](../preparation/ragtruth-data.md).

The frozen campaigns are:

| Pooling | Confirmatory signature | Configuration |
| --- | --- | --- |
| Gated Attention | `4e12933c51136624` | `ragtruth_lora_attention_mil_confirmatory.yaml` |
| Mean | `6bb8a5d4e8041215` | `ragtruth_lora_mean_mil_confirmatory.yaml` |
| Max | `53f48ce4154f075e` | `ragtruth_lora_max_mil_confirmatory.yaml` |

All campaigns use dataset signature `0cdf598fa866741d`, split signature
`525edec2966a4fac`, split seed 42, and model seeds 0, 1, and 2. Their frozen
protocol payloads are identical after removing the operational run name and
pooling identity.

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

Use `--resume` with the training phase after an interruption. Each pooling type
has a distinct signature, so it cannot reuse or overwrite an incompatible
campaign directory.

## Outputs

Each campaign writes `runs/ragtruth_confirmatory/<signature>/` with one
directory per seed, validation predictions and thresholds, RAGTruth test
predictions, PublicHearingBR zero-shot predictions, grouped bootstrap summaries,
resolved configuration, frozen protocol, and manifests.

The implementation is covered by `tests/test_model_pooling.py` and
`tests/test_ragtruth_zero_shot.py`. The tests verify valid-slot Mean and
feature-wise Max semantics, masks in different positions and cardinalities,
negative valid features for Max, the direct `[B,128]` pre-classifier pooling
shape, Gated Attention regression behavior, pooling-specific signatures, and
Mean/Max zero-shot source validation.

## Result scope

The ablation uses three fixed seeds and reports their mean and sample standard
deviation. Thresholds are selected separately per pooling and seed on RAGTruth
validation only, then frozen before RAGTruth test and PublicHearingBR zero-shot
evaluation.

The existing paired grouped bootstraps compare Gated Attention with the
off-the-shelf NLI baseline; they do not estimate Attention-versus-Mean or
Attention-versus-Max uncertainty. Pooling comparisons are therefore descriptive.

`notebooks/03_pooling_ablation.ipynb` is the read-only visual summary of these
frozen artifacts. It uses `notebooks/configs/pooling_ablation.local.yaml` or
the `NOTEBOOK_CONFIG` environment variable to select campaign paths.
