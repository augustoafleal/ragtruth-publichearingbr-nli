# Off the shelf NLI baseline

## Purpose

This experiment evaluates a fixed multilingual NLI model on PublicHearingBR.
It does not train a model or load LoRA adapters. The hallucination score is one
minus the maximum entailment probability across valid evidence chunks.

## Required preparation

Prepare or locate the PublicHearingBR dataset. See
[PublicHearingBR data](../preparation/publichearingbr-data.md).

The configuration fixes the mDeBERTa model revision and the dataset revision.

## Run

Validate the run contract without loading the model:

```bash
python scripts/evaluate_publichearing_off_the_shelf_nli.py \
  --config configs/publichearing_off_the_shelf_max_entailment.yaml \
  --validate-only
```

Run the evaluation:

```bash
python scripts/evaluate_publichearing_off_the_shelf_nli.py \
  --config configs/publichearing_off_the_shelf_max_entailment.yaml
```

## Outputs

The run creates `runs/publichearing_off_the_shelf_max_entailment/<signature>/`.
It contains predictions, threshold-free metrics, integrity checks, configuration,
manifest, and report.

## Result scope

This is the fixed baseline used by the transfer and paired bootstrap analyses.
