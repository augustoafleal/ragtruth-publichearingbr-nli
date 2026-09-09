# CI do reporting baseado em artefatos congelados

Os testes de reporting que validam provenance e resultados experimentais reais
usam o marker `frozen_artifacts`. Eles não são executados pelo CI padrão,
porque `runs/` e `results/` não são versionados no repositório.

## Auditoria

Os seguintes testes dependem dos artefatos congelados de reporting:

- `tests/test_paper_reporting.py`
  - `test_official_provenance_and_superseded_rejection`
  - `test_main_tables_thresholded_invariants_and_forest_smoke`
  - `test_full_reporting_registry_and_repeatability`
- `tests/test_thresholded_attention_set_analysis.py`
  - `test_attention_and_set_campaigns_are_accepted_by_preflight`
  - `test_small_real_bootstrap_is_deterministic_and_thresholds_are_fixed`

Os testes de lógica desses módulos que usam apenas DataFrames pequenos,
configurações ou diretórios temporários permanecem autocontidos.

## Execução

O CI padrão executa:

```bash
pytest -q -m "not frozen_artifacts"
```

A trilha de integração deve executar, após restaurar o bundle imutável:

```bash
pytest -q -m frozen_artifacts
```

Os testes continuam chamando `validate_sources()` e os validadores de
provenance de forma estrita. A ausência de artefatos nessa segunda trilha deve
ser reportada como erro.

## Artefatos mínimos para a futura trilha

Para `test_paper_reporting.py`, o bundle precisa restaurar:

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

Para `test_thresholded_attention_set_analysis.py`, também são necessários os
artefatos das campanhas:

- `runs/ragtruth_pt_nllb_confirmatory/70bb1cce59b8c824/`
- `runs/ragtruth_pt_nllb_set_transformer_confirmatory/29ec472694bb61a8/`

Em cada campanha, devem estar disponíveis o manifesto e a configuração
resolvida, os manifests das seeds `0`, `1` e `2`, seus `validation_predictions.csv`
e `thresholds.json`, o `aggregate/per_seed_metrics.csv` e, para cada seed, o
manifesto, `metrics.json` e `predictions.parquet` do artefato
`publichearing_zero_shot`.

O job futuro pode extrair o bundle diretamente nesses caminhos relativos ao
checkout. Uma variável como `REPORTING_ARTIFACT_ROOT` só será necessária se o
bundle for mantido fora da árvore do repositório; nesta etapa não há download
nem alteração do comportamento de `build_source_registry()`.
