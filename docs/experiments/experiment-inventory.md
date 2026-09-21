# Experiment inventory

This inventory lists the six core scientific experiment families and the
controlled pooling ablation supported by this repository. Smoke runs, screening
runs, preliminary diagnostics, and data preparation runs are retained for
traceability but are not primary results.

## Included experiments

- **PublicHearingBR supervised**
  `results/publichearing_lora_attention_mil/741ed3152c2175e7/` is the in-domain
  LoRA reference: AUPRC 0.6044, AUROC 0.8851, Brier 0.0789.
- **RAGTruth confirmatory**
  `runs/ragtruth_confirmatory/4e12933c51136624/` is the source-domain campaign.
  Across 3 seeds: RAGTruth test — AUPRC 0.5273 ± 0.0276, AUROC 0.9009 ±
  0.0066, Brier 0.0736 ± 0.0048; PublicHearingBR zero-shot — AUPRC 0.5949 ±
  0.0142, AUROC 0.8788 ± 0.0036, Brier 0.0881 ± 0.0075.
- **Controlled pooling ablation**
  The same RAGTruth confirmatory protocol compares Gated Attention
  (`4e12933c51136624`), Mean (`6bb8a5d4e8041215`), and Max
  (`53f48ce4154f075e`).

  | PublicHearingBR zero-shot, three-seed mean ± SD | AUPRC | AUROC | Brier |
  | --- | ---: | ---: | ---: |
  | Gated Attention | **0.5949 ± 0.0142** | 0.8788 ± 0.0036 | 0.0881 ± 0.0075 |
  | Mean | 0.5735 ± 0.0036 | 0.8661 ± 0.0067 | 0.0956 ± 0.0151 |
  | Max | 0.5704 ± 0.0188 | **0.8840 ± 0.0040** | **0.0864 ± 0.0103** |

  Attention is retained because it has the highest mean target AUPRC, the
  decision criterion used for the imbalanced target. Max has the strongest
  AUROC/Brier profile. This is a descriptive three-seed comparison, not a
  pairwise bootstrap superiority test.
- **Off the shelf NLI baseline**
  `runs/publichearing_off_the_shelf_max_entailment/54d9c623f8685c39/` is the
  fixed target-domain comparator: AUPRC 0.3375, AUROC 0.7680, Brier 0.1143
  (diagnóstico de qualidade do score).
- **Threshold transfer**
  `runs/ragtruth_off_the_shelf_threshold_transfer/3ceffc4a74b484fe/` transfers
  frozen operating points. `best_f1`: baseline F1/recall/FPR 0.1570/0.0918/
  0.0104; LoRA 0.5444 ± 0.0078 / 0.5236 ± 0.0686 / 0.0535 ± 0.0218.
  `fpr10`: baseline 0.1520/0.0878/0.0091; LoRA 0.5425 ± 0.0063 / 0.5995 ±
  0.0321 / 0.0820 ± 0.0138.
- **Continuous paired bootstrap**
  `runs/publichearing_paired_grouped_bootstrap/ae19afe0e45715ae/` estimates
  ranking-metric uncertainty: ΔAUPRC +0.2575 (IC95% [0.2158, 0.2946]),
  ΔAUROC +0.1109 (IC95% [0.0873, 0.1341]) and Brier improvement +0.0262
  (IC95% [0.0183, 0.0346]).
- **Thresholded paired bootstrap**
  `runs/publichearing_thresholded_paired_bootstrap/959b3fe48236fb73/` estimates
  operational-metric uncertainty. `best_f1`: ΔF1 +0.3874 [0.3417, 0.4317],
  Δrecall +0.4318 [0.3886, 0.4732], ΔMCC +0.3030 [0.2527, 0.3529], Δbalanced
  accuracy +0.1944 [0.1724, 0.2154]. `fpr10`: ΔF1 +0.3905 [0.3443, 0.4343],
  Δrecall +0.5116 [0.4696, 0.5508], ΔMCC +0.2888 [0.2390, 0.3386], Δbalanced
  accuracy +0.2194 [0.1976, 0.2396]. All intervals are IC95%.

Each `<signature>` is generated from the protocol and its frozen inputs. Use
the manifest in the generated directory to verify the result.

The continuous bootstrap has one historical duplicate:
`111fb4f94b3e2c14` is scientifically and numerically equivalent to canonical
`ae19afe0e45715ae`. The latter is canonical because its recorded implementation
version is `paired-grouped-bootstrap-v2`; the earlier materialization records
`v1` and lacks only added provenance diagnostics.

Each experiment page explains the purpose, required preparation, command, and
outputs.

## Additional reproducibility analysis

- **Offline LLM tokenization and current-equivalent cost analysis**
  reconstructs the published prompt configurations locally, counts tokens with
  recorded tokenizer mappings, and applies versioned current USD pricing to the
  four LLM comparators. It also reports the cloud-equivalent inference cost of
  one Set Transformer model on a Tesla T4. See
  [Offline LLM cost analysis](llm-cost-analysis.md). This is a cost analysis,
  not a new model-training campaign, and it does not modify the frozen
  experiment artifacts.
