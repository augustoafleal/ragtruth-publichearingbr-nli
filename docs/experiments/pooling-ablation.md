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

## Frozen campaigns

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

## Outputs and checks

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

## Results

Threshold-free metrics are mean ± sample standard deviation across the three
seeds. AUPRC is the decision criterion for the imbalanced PublicHearingBR target;
AUROC and Brier score are complementary diagnostics.

| Dataset | Pooling | AUPRC | AUROC | Brier (lower is better) |
| --- | --- | ---: | ---: | ---: |
| RAGTruth test | Gated Attention | 0.5273 ± 0.0276 | 0.9009 ± 0.0066 | 0.0736 ± 0.0048 |
| RAGTruth test | Mean | **0.5366 ± 0.0122** | 0.8851 ± 0.0108 | 0.0738 ± 0.0125 |
| RAGTruth test | Max | 0.5230 ± 0.0288 | **0.9028 ± 0.0044** | **0.0684 ± 0.0138** |
| PublicHearingBR zero-shot | Gated Attention | **0.5949 ± 0.0142** | 0.8788 ± 0.0036 | 0.0881 ± 0.0075 |
| PublicHearingBR zero-shot | Mean | 0.5735 ± 0.0036 | 0.8661 ± 0.0067 | 0.0956 ± 0.0151 |
| PublicHearingBR zero-shot | Max | 0.5704 ± 0.0188 | **0.8840 ± 0.0040** | **0.0864 ± 0.0103** |

For PublicHearingBR operating points selected only on RAGTruth validation:

| Regime | Pooling | F1 | Recall | FPR |
| --- | --- | ---: | ---: | ---: |
| `best_f1` | Gated Attention | **0.5444 ± 0.0078** | **0.5236 ± 0.0686** | 0.0535 ± 0.0218 |
| `best_f1` | Mean | 0.5013 ± 0.0098 | 0.4165 ± 0.0208 | **0.0329 ± 0.0054** |
| `best_f1` | Max | 0.5373 ± 0.0127 | 0.4963 ± 0.0446 | 0.0469 ± 0.0107 |
| `fpr10` | Gated Attention | 0.5425 ± 0.0063 | 0.5995 ± 0.0321 | 0.0820 ± 0.0138 |
| `fpr10` | Mean | 0.5313 ± 0.0073 | **0.6148 ± 0.0158** | 0.0938 ± 0.0034 |
| `fpr10` | Max | **0.5455 ± 0.0110** | 0.6028 ± 0.0405 | **0.0816 ± 0.0164** |

## Interpretation and scope

Gated Attention is retained as the selected aggregation because it has the
highest mean zero-shot PublicHearingBR AUPRC, the decision criterion used for
the imbalanced target. Max is a credible alternative with the strongest target
AUROC and Brier profile, but it does not exceed Attention on target AUPRC. Mean
does not lead the target threshold-free metrics.

This is a descriptive three-seed ablation. The existing paired grouped
bootstraps compare Gated Attention with the off-the-shelf NLI baseline; they do
not estimate uncertainty for Attention-versus-Mean or Attention-versus-Max.
They must not be interpreted as pairwise pooling superiority tests.

`notebooks/03_pooling_ablation.ipynb` is the read-only visual summary of these
frozen artifacts. It uses `notebooks/configs/pooling_ablation.local.yaml` or
the `NOTEBOOK_CONFIG` environment variable to select campaign paths.
