# Confirmatory paper figures without MADLAD

## Purpose

This presentation-only analysis regenerates the three confirmatory paper
figures using the current NLLB-only scope. It does not modify the historical
five-condition reporting mode and does not recalculate bootstrap statistics,
metrics, training, inference, or experimental results.

The MADLAD artifacts remain available for historical analyses, but MADLAD is
not displayed in these three confirmatory figures.

## Frozen source

The figures read the existing frozen statistical summary:

`runs/publichearing_final_paired_bootstrap/fc7bbae7bd5d0c99/final_bootstrap_summary.csv`

The isolated implementation is
`scripts/regenerate_confirmatory_paper_figures.py`. It reuses the source
registry, frozen-source validation, point extraction, contrast selection, and
plotting helpers from `src/ragtruth_transfer/paper_reporting.py`.

## Confirmatory scope

The three displayed conditions are:

| Condition | Interpretation |
| --- | --- |
| `EN → PH PT` | Direct transfer |
| `PT-NLLB → PH PT` | Translate-train with NLLB |
| `EN → PH EN-NLLB` | Translate-test with NLLB |

The translation-effects forest contains only:

- NLLB training-side translation;
- NLLB target-side translation.

MADLAD is excluded by a local selection in the isolated script. The historical
`CONDITIONS` and `TRANSLATION_SPECS` definitions remain unchanged.

## Run

Run the isolated confirmatory figure generation:

```bash
python scripts/regenerate_confirmatory_paper_figures.py
```

The script validates the frozen numerical values before writing any figure. It
updates only the three target PNGs and their three corresponding inventory
entries.

The translation-effects forest uses the compact height `1.6` so that removing
the two MADLAD effects does not leave the vertical space associated with the
historical four-effect layout. The other figures retain the existing plotting
style and default dimensions.

## Outputs

The script regenerates:

- `results/paper/figures/figure_main_auprc_comparison.png`;
- `results/paper/figures/figure_set_vs_attention_auprc_forest.png`;
- `results/paper/figures/figure_translation_effects_auprc_forest.png`.

It updates only the matching rows in
`results/paper/figure_inventory.csv`. Tables, claims, manifests, frozen CSVs,
and other figures are not regenerated.

## Validation

Run the dedicated test together with the historical reporting tests:

```bash
pytest -q tests/test_confirmatory_paper_figures.py tests/test_paper_reporting.py
```

The dedicated test checks the three-condition selection, the two NLLB
translation effects, the expected frozen values, the absence of MADLAD labels,
the target-only output set, and preservation of unrelated inventory rows.

## Historical mode

The complete reporting generator remains:

```bash
python scripts/generate_paper_results.py
```

Its default five-condition behavior is preserved. The isolated confirmatory
script is opt-in and does not replace the historical reporting generator.
