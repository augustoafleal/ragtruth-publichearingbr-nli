# PublicHearingBR supervised confirmatory experiment

## Purpose

This experiment trains LoRA with attention MIL directly on PublicHearingBR. It
uses grouped cross validation by hearing and three fixed seeds. It is the main
in domain reference.

## Required preparation

Prepare or locate the PublicHearingBR dataset first. See
[PublicHearingBR data](../preparation/publichearingbr-data.md).

The confirmatory configuration fixes the dataset revision, model revision,
five outer folds, five inner folds, and seeds 101, 202, and 303.

## Run

Run the full confirmatory campaign:

```bash
python scripts/run_publichearing_cv.py \
  --config configs/publichearing_lora_attention_mil_confirmatory.yaml
```

Aggregate completed folds and seeds:

```bash
python -m ragtruth_transfer.publichearing.cli aggregate \
  --config configs/publichearing_lora_attention_mil_confirmatory.yaml
```

Validate a completed run:

```bash
python -m ragtruth_transfer.publichearing.cli validate-run \
  --run-dir results/publichearing_lora_attention_mil/<signature>
```

## Outputs

The run creates `results/publichearing_lora_attention_mil/<signature>/`.
Important outputs include fold assignments, predictions, OOF metrics, grouped
bootstrap results, configuration, signature, manifest, and archive.

## Result scope

Use the completed confirmatory run under
`results/publichearing_lora_attention_mil/<signature>/` for interpretation.
The screening run and smoke run are development artifacts.
