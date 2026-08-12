# RAGTruth data preparation

## Purpose

Create the signed Parquet training view used by the RAGTruth confirmatory
experiment.

## Source data

Download the RAGTruth raw files first:

```bash
python scripts/download_ragtruth.py --output-dir data/raw/ragtruth
```

## Build the Top 4 dataset

Run the label independent Top 4 preparation on a GPU node:

```bash
python scripts/prepare_ragtruth_top4.py \
  --config configs/ragtruth_qa_top4_label_independent.yaml \
  --device cuda \
  --batch-size 64 \
  --resume
```

## Build the training view

Use the signature produced by the prior step:

```bash
python scripts/build_ragtruth_training_view.py \
  --input-run-dir results/ragtruth_qa_top4_label_independent/ragtruth_qa_top4_label_independent_embeddings/<signature> \
  --output-root results/ragtruth_qa_training_view
```

## Outputs

The confirmatory campaign expects the deduplicated Parquet dataset and manifest
under `results/ragtruth_qa_training_view/<policy>/<signature>/`. Verify the
expected signature in the confirmatory configuration before training.
