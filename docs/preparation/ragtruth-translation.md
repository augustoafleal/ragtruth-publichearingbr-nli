# RAGTruth translation preparation

## Purpose

Create Portuguese variants of the processed RAGTruth JSONL dataset with the
same translation pipeline used for both NLLB and MADLAD. This is a data
preparation step, not a scientific experiment or a model-training campaign.

The pipeline translates only `claim` and evidence entries whose corresponding
`evidence_mask` value is `true`. It preserves the schema, metadata, masks,
splits, evidence slots, and record order.

## Requirements

Install the project environment and run this step on a GPU node when possible:

```bash
python -m pip install -e ".[dev]"
```

The model weights are downloaded by Transformers on first use. No weights are
needed for the unit tests.

## Smoke test

Run the smoke configs before a larger cluster campaign. They use the real
dataset but process at most three records from each split and write to separate
directories:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_nllb_smoke.yaml

python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_madlad_smoke.yaml
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

MADLAD with a deterministic 25% sample of the dataset:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_madlad_sample25.yaml
```

This output is written separately to
`data/processed/ragtruth_textual_pt_madlad_sample25/`.

The normal configs set `data.sample_fraction: 1.0` and
`data.sample_seed: 42`, which processes every record in each configured split.
The outputs are:

- `data/processed/ragtruth_textual_pt_nllb/`
- `data/processed/ragtruth_textual_pt_madlad/`

## Deterministic sampling

To translate a reproducible fraction without changing code, set for example:

```yaml
data:
  sample_fraction: 0.25
  sample_seed: 42
```

Sampling is performed independently for `train`, `validation`, and `test`.
Selected records retain their original order. `sample_fraction` must be in
`(0, 1]`; `1.0` means the full dataset.

Smoke and sampling are mutually exclusive. Use `max_examples_per_split` only
for smoke configs and `sample_fraction` for partial or full campaigns.

## Cache and resume

Each output has a persistent SQLite translation cache. Completed text
translations can be reused after an interruption. The cache is checked against
the model, language, generation parameters, and input hashes. The output
manifest additionally records the mode, fraction, seed, selection, and counts
processed per split, so smoke, partial, and full outputs remain distinguishable.

To resume explicitly with the same configuration, rerun the entrypoint with
`--resume`:

```bash
python scripts/translate_ragtruth.py \
  --config configs/ragtruth_translate_nllb.yaml \
  --resume
```

The same `--resume` option works with the NLLB normal config, the MADLAD normal
config, and either smoke config. A changed sampling fraction, seed, model, or
input dataset is treated as a different execution and cannot reuse an
incompatible output manifest.
