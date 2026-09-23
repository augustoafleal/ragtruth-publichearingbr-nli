# Reporting CI based on frozen artifacts

Reporting tests that validate provenance and real experimental results use the
`frozen_artifacts` marker. They are not executed by the standard CI suite
because `runs/` and `results/` are not versioned in the repository.

## Audit

The following tests depend on frozen reporting artifacts:

- `tests/test_paper_reporting.py`
  - `test_official_provenance_and_superseded_rejection`
  - `test_main_tables_thresholded_invariants_and_forest_smoke`
  - `test_full_reporting_registry_and_repeatability`
- `tests/test_confirmatory_paper_figures.py`
  - `test_confirmatory_figures_exclude_madlad_and_preserve_other_inventory`
- `tests/test_thresholded_attention_set_analysis.py`
  - `test_attention_and_set_campaigns_are_accepted_by_preflight`
  - `test_small_real_bootstrap_is_deterministic_and_thresholds_are_fixed`

Logic tests in these modules that use only small DataFrames, configurations, or
temporary directories remain self-contained.

## Execution

Standard CI runs:

```bash
pytest -q -m "not frozen_artifacts"
```

The integration track should run after restoring the immutable bundle:

```bash
pytest -q -m frozen_artifacts
```

The tests continue to call `validate_sources()` and the provenance validators
strictly. Missing artifacts in this second track must be reported as an error.

## Minimum artifacts for the future track

For `test_paper_reporting.py` and
`test_confirmatory_paper_figures.py`, the bundle must restore:

- `runs/publichearing_final_paired_bootstrap/fc7bbae7bd5d0c99/`
  - `final_bootstrap_summary.csv`
  - `final_bootstrap_report.md`
  - `manifest.json`
  - `resolved_config.json`
  - `population_validation.json`
- `runs/publichearing_pooling_paired_bootstrap/b92d4fd6a6a7e771/observed_metrics.json`
- `runs/publichearing_off_the_shelf_max_entailment/54d9c623f8685c39/metrics.json`
- `runs/ragtruth_off_the_shelf_threshold_transfer/3ceffc4a74b484fe/publichearing_metrics.json`
- `results/publichearing_lora_attention_mil/741ed3152c2175e7/outputs/overall_oof_metrics.csv`
- `runs/ragtruth_pt_nllb_filtered_confirmatory/63745412afdb52ac/aggregate/aggregate_metrics.csv`
- `runs/ragtruth_pt_nllb_bertimbau_confirmatory/fce6272e2e72726d/aggregate/aggregate_metrics.csv`
- `runs/publichearing_pt_nllb_attention_vs_set_thresholded_bootstrap/e0c75270065fc471/bootstrap_summary.json`
- `results/paper/figure_inventory.csv`

For `test_thresholded_attention_set_analysis.py`, the campaign artifacts are
also required:

- `runs/ragtruth_pt_nllb_confirmatory/70bb1cce59b8c824/`
- `runs/ragtruth_pt_nllb_set_transformer_confirmatory/29ec472694bb61a8/`

Each campaign must provide its manifest and resolved configuration, the
manifests for seeds `0`, `1`, and `2`, their `validation_predictions.csv` and
`thresholds.json`, `aggregate/per_seed_metrics.csv`, and, for each seed, the
manifest, `metrics.json`, and `predictions.parquet` for the
`publichearing_zero_shot` artifact.

The future job can extract the bundle directly into these paths relative to the
checkout. A variable such as `REPORTING_ARTIFACT_ROOT` is needed only if the
bundle is kept outside the repository tree; this setup performs no download and
does not change `build_source_registry()` behavior.
