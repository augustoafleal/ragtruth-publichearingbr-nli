# RAGTruth translation preparation

## Purpose

Both NLLB and MADLAD create separate Portuguese Parquets from the canonical
English post-RAG, post-deduplication training view. This is data preparation,
not training.

The NLLB pipeline translates only `claim` and chunks whose corresponding
`evidence_mask` value is `true`. It preserves every other column, schema,
metadata, masks, splits, evidence slots, and record order. It never falls back
to the English source when a translation is missing.

## Requirements

Install the project environment and run this step on a GPU node when possible:

```bash
python -m pip install -e ".[dev]"
```

The model weights are downloaded by Transformers on first use. No weights are
needed for the unit tests.

## Smoke test

Validate the NLLB source and field contract without loading a model:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_nllb_smoke.yaml --validate-only

python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_madlad_smoke.yaml --validate-only
```

The smoke output manifests use `mode: smoke` and
`max_examples_per_split: 3`. They must not be used as full translated
datasets.

## Full translation

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

MADLAD uses the same generation limits, beam count and batch sizes as the
corresponding NLLB configs, while retaining its own model and language prefix.

The normal configs process every canonical Parquet row. Their outputs are:

- `data/processed/ragtruth_confirmatory_pt_nllb/dataset.parquet`
- `data/processed/ragtruth_confirmatory_pt_madlad/dataset.parquet`

## Cache and resume

Each output has a persistent SQLite translation cache. Completed text
translations can be reused after an interruption. The cache is checked against
the model, language, generation parameters, and input hashes. The output
manifest additionally records backend settings, source/output hashes, cache
usage, QA diagnostics and counts, so smoke and full outputs remain distinguishable.

To resume explicitly with the same configuration, rerun the entrypoint with
`--resume`:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_nllb.yaml \
  --resume
```

The same `--resume` option works with the NLLB normal config, the MADLAD normal
config, and either smoke config. A changed backend, model, generation settings,
or input dataset is treated as a different execution and cannot reuse an
incompatible output manifest.
