# RAGTruth → PublicHearingBR-NLI

Este repositório implementa a preparação de dados, os seis protocolos
científicos centrais e uma ablação controlada de pooling do estudo RAGTruth →
PublicHearingBR:

1. PublicHearingBR supervisionado e in-domain
2. RAGTruth confirmatório com três seeds
3. baseline NLI off-the-shelf
4. transferência de thresholds do RAGTruth
5. bootstrap pareado e agrupado para scores contínuos
6. bootstrap pareado e agrupado para métricas thresholded

A ablação compara `gated_attention`, `mean`, `max` e `set_transformer` no mesmo protocolo
RAGTruth → PublicHearingBR; ela é documentada em
`docs/experiments/pooling-ablation.md`.

The inferential Gated Attention × Set Transformer comparison is documented in
`docs/experiments/set-transformer-paired-bootstrap.md` and uses only existing
zero-shot prediction Parquets.

Smoke tests, screening e preparação de dados não são resultados científicos
principais.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

## Documentação

A documentação completa descreve os dados necessários, cada protocolo,
comandos e outputs:

```bash
mkdocs serve
```

Para validar a documentação:

```bash
mkdocs build --strict
```

## Preparação do RAGTruth

Baixe os arquivos de origem:

```bash
python scripts/download_ragtruth.py --output-dir data/raw/ragtruth
```

Construa o Top-4 semântico com boundaries independentes dos labels:

```bash
python scripts/prepare_ragtruth_top4.py \
  --config configs/ragtruth_qa_top4_label_independent.yaml \
  --device cuda \
  --batch-size 64 \
  --resume
```

Construa a visão Parquet deduplicada:

```bash
python scripts/build_ragtruth_training_view.py \
  --input-run-dir results/ragtruth_qa_top4_label_independent/ragtruth_qa_top4_label_independent_embeddings/<signature> \
  --output-root results/ragtruth_qa_training_view
```

## Tradução do RAGTruth para português

O pipeline de tradução lê os JSONL processados em
`data/processed/ragtruth_textual` e mantém os mesmos splits, ordem e schema.
Somente `claim` e as evidências com `evidence_mask: true` são traduzidos; os
demais campos e slots mascarados são preservados.

NLLB:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_nllb.yaml
```

MADLAD:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_madlad.yaml
```

Antes de uma campanha no cluster, há smoke configs que processam no máximo
3 registros por split:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_nllb_smoke.yaml

python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_madlad_smoke.yaml
```

As configs normais usam `sample_fraction: 1.0` (100%) e `sample_seed: 42`.
Para uma amostra determinística, altere a fração para, por exemplo,
`0.25` (25% de cada split). Smoke e sampling são opções distintas e não
podem ser configurados juntos; cada smoke output usa um diretório separado.

Os outputs são, respectivamente, `data/processed/ragtruth_textual_pt_nllb`
e `data/processed/ragtruth_textual_pt_madlad`. O device é escolhido por
`device: auto` (CUDA quando disponível, caso contrário CPU); ele pode ser
alterado para `cpu` ou `cuda` no YAML. Um cache SQLite persistente fica junto
ao output e é reutilizado automaticamente após interrupções. Ele é validado
contra o modelo, parâmetros de geração e hashes dos inputs, evitando misturas
entre configurações incompatíveis. O `manifest.json` registra a configuração,
splits e contagens da execução concluída.

## Experimentos canônicos

### PublicHearingBR supervisionado

```bash
python scripts/run_publichearing_cv.py \
  --config configs/publichearing_lora_attention_mil_confirmatory.yaml
```

### RAGTruth confirmatório

Valide os inputs e execute as três fases:

```bash
python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --validate-only

python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --phase train

python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --phase evaluate

python scripts/run_ragtruth_confirmatory.py \
  --config configs/ragtruth_lora_attention_mil_confirmatory.yaml \
  --phase aggregate
```

### Ablação de pooling

As variantes Mean, Max e Set Transformer usam, respectivamente,
`configs/ragtruth_lora_mean_mil_confirmatory.yaml`,
`configs/ragtruth_lora_max_mil_confirmatory.yaml` e
`configs/ragtruth_lora_set_transformer_mil_confirmatory.yaml`. Execute
`--validate-only`, `--phase train`, `--phase evaluate` e `--phase aggregate`
para cada variante. As signatures e resultados canônicos estão em
`docs/experiments/pooling-ablation.md`.

### Baseline NLI off-the-shelf

```bash
python scripts/evaluate_publichearing_off_the_shelf_nli.py \
  --config configs/publichearing_off_the_shelf_max_entailment.yaml
```

### Transferência de thresholds

```bash
python scripts/run_ragtruth_off_the_shelf_threshold_transfer.py \
  --config configs/ragtruth_off_the_shelf_threshold_transfer.yaml
```

### Bootstrap contínuo

```bash
python scripts/run_publichearing_paired_grouped_bootstrap.py \
  --config configs/publichearing_paired_grouped_bootstrap.yaml
```

### Bootstrap thresholded

```bash
python scripts/run_publichearing_thresholded_paired_bootstrap.py \
  --config configs/publichearing_thresholded_paired_bootstrap.yaml
```

## Validação local

```bash
python -m compileall -q src scripts
pytest -q
mkdocs build --strict
```

Os artefatos científicos são gravados em `runs/` e `results/`. Manifests,
resolved configs, hashes, predictions e métricas nesses diretórios são a fonte
de verdade para cada execução congelada.
