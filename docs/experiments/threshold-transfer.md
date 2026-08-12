# Threshold transfer from RAGTruth to PublicHearingBR

## Purpose

This experiment transfers frozen operating thresholds from RAGTruth validation
to PublicHearingBR. It compares the off the shelf baseline and the three
RAGTruth LoRA seeds at `best_f1` and `fpr10` operating points.

## Required preparation

Complete the [RAGTruth confirmatory experiment](ragtruth-confirmatory.md), the
[off the shelf NLI baseline](off-the-shelf-nli.md), and prepare both datasets.
See [RAGTruth data](../preparation/ragtruth-data.md) and
[PublicHearingBR data](../preparation/publichearingbr-data.md).

## Run

Validate the provenance and frozen inputs first:

```bash
python scripts/run_ragtruth_off_the_shelf_threshold_transfer.py \
  --config configs/ragtruth_off_the_shelf_threshold_transfer.yaml \
  --validate-only
```

Run the transfer evaluation:

```bash
python scripts/run_ragtruth_off_the_shelf_threshold_transfer.py \
  --config configs/ragtruth_off_the_shelf_threshold_transfer.yaml
```

## Outputs

The run creates `runs/ragtruth_off_the_shelf_threshold_transfer/<signature>/`.
It stores frozen thresholds, source and target metrics, comparisons, integrity
audit, configuration, predictions, manifest, and report.

## Result scope

Threshold selection uses RAGTruth validation only. PublicHearingBR labels are
used only after inference to calculate target metrics.
