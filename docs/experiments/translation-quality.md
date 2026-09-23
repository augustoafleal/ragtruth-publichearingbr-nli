# Translation-quality evaluation

## Purpose

This framework measures how faithful the machine translations of RAGTruth
and PublicHearingBR (NLLB and MADLAD) are and relates that quality, descriptively, to the
hallucination detector's performance. It tests the hypothesis that **poor
machine translation degrades hallucination detection** without over-claiming:
all Stage 2 comparisons are descriptive.

It is organised in three signature-frozen stages, mirroring the repository's
conventions (frozen dataclass YAML config, output directory keyed by a
`sha256[:16]` recipe signature, `manifest.json` + `resolved_config.json`,
atomic writes, `--validate-only`, Portuguese error messages).

Translation quality is computed **once per translation backend** and has **no
seed**; seeds only appear downstream in Stage 2, when the same quality artifact
is related to different detector runs.

## Stage 0 — source→target integrity and alignment

The alignment contract is direction-agnostic. Each configuration declares
`source_language`, `target_language`, `source_artifact` and
`translated_artifact`; RAGTruth keeps its historical `en_parquet`/`pt_parquet`
aliases. NLI deltas are always `target - source`, so a positive
`nli_entail_delta` means that entailment probability increased after
translation, regardless of the language pair.

The Portuguese datasets overwrite only the `claim`/`chunk_*` text; every offset
and provenance column is preserved. The canonical English training view

```
results/ragtruth_qa_training_view/deduplicate_source_claim_drop_conflicts_v1/0cdf598fa866741d/dataset.parquet
```

is the exact input that was translated, and shares 100 % of its `example_id`
with the PT datasets. Stage 0 therefore **joins the two artifacts** instead of
reconstructing English from `data/raw/ragtruth` or re-running the
tokenizer/chunker.

It validates, and fails loudly on any inconsistency:

- coverage (`example_id` of PT ⊆ EN) and uniqueness;
- `source_id`, `label`, `split`, `response_id`;
- `evidence_mask`;
- per-chunk provenance (`source_index`, `window_index`, `token_start`,
  `token_end`, `sha256`);
- the EN chunk text against its stored `chunk_k_sha256`.

```bash
python scripts/align_translation_quality.py --config configs/translation_quality_nllb.yaml --validate-only
python scripts/align_translation_quality.py --config configs/translation_quality_nllb.yaml
```

Output: `aligned.parquet` + `manifest.json` under
`results/translation_quality/ragtruth_pt_<backend>/alignment/<sig16>/`.
On the full data the gate reports 34,604 rows, 104,252 valid chunks and 100 %
SHA-256 match.

PublicHearingBR uses its canonical nested JSONL directly; it is not converted
to a RAGTruth Parquet schema. The strict join uses
`hearing_id:person_index:opinion_index`, labels and all non-translated
metadata, with 4,235 modelable examples and four evidence chunks per example:

```bash
python scripts/align_translation_quality.py --config configs/translation_quality_publichearing_nllb.yaml --validate-only
python scripts/align_translation_quality.py --config configs/translation_quality_publichearing_madlad.yaml --validate-only
```

## Stage 1 — quality scoring

Per segment (the claim and each valid evidence chunk) the selected adapters run.
Heavy adapters are imported lazily, so the core install and the test suite need
none of them (`pip install -e '.[quality]'` adds COMETKiwi).

- **Heuristics** — reuses `translation_qa.classify_translation_pair` (length
  ratio, repetition severity, empty/identical/control-char flags). No deps.
- **COMETKiwi** — reference-free QE (`Unbabel/wmt22-cometkiwi-da`), CPU/GPU via
  `device`, configurable `batch_size`. The model revision is pinned and resolved
  with `huggingface_hub.snapshot_download(revision=...)`, so the weights do not
  depend on the Hub's current state.
- **NLI-consistency** — `P(entail | premise=evidence, hypothesis=claim)` scored
  in source and target with a plain XNLI model (independent of the trained detector
  checkpoint), emitting `nli_entail_delta` (`target - source`),
  `nli_abs_delta`/`nli_entail_abs_delta`, and `nli_label_agree`. The model and tokenizer are loaded with an explicit
  `revision`.
- **Detector truncation** (diagnostic only) — whether PT expansion overflows the
  detector's `max_length` when EN did not (`detector_truncation_introduced`).
  It is independent of the NLI scorer: enable `scoring.detector_truncation` and
  it runs with any metric set (e.g. `metrics: [heuristics, cometkiwi]`), using
  only the pinned detector tokenizer.

Aggregation produces segment-level and example-level tables. **No arbitrary
composite score is created**: the heuristics, COMETKiwi, NLI-consistency and
truncation signals are preserved individually (`cometkiwi_chunk_min`,
`nli_abs_delta_max`, `nli_any_flip`, `any_truncation_introduced`, ...).

```bash
python scripts/score_translation_quality.py --config configs/translation_quality_nllb.yaml --validate-only
python scripts/score_translation_quality.py --config configs/translation_quality_nllb.yaml
```

Output: `segment_scores.parquet`, `example_scores.parquet`,
`quality_summary.json`, `manifest.json`.

### Pinned model revisions

The models used by Stage 1 are pinned to immutable Hub revisions (recorded in
the config, the scoring signature and the manifest):

| Model | Revision |
|---|---|
| `Unbabel/wmt22-cometkiwi-da` | `1ad785194e391eebc6c53e2d0776cada8f83179a` |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `b5113eb38ab63efdd7f280f8c144ea8b13f978ce` |

The detector tokenizer used for the truncation diagnostic is pinned to the same
mDeBERTa revision.

## Stage 2 — linkage with the detector

Joins `example_scores.parquet` to the existing confirmatory
`predictions.parquet` with **strict validation**: predictions must carry
`example_id`, `score`, `source_id` and `label` (failure is explicit if any is
missing); coverage (every prediction must have a score), uniqueness,
`example_id`, `source_id`, `label`, translation backend and the protocol
signature are all checked. A silent inner merge with loss of examples is
rejected. Backend/protocol are validated against the confirmatory run metadata
(`frozen_protocol.json` next to the predictions), falling back to path
inspection when that metadata is unavailable.

The operating threshold is **consumed** from the protocol artifact
(`thresholds.json` / `thresholds_applied.json`); it is never re-selected on the
test pool. The analysis reports per-example error (`abs_error`, `log_loss`,
`correct`), detection metrics stratified by quality quantile with grouped
bootstrap CIs over `source_id`, correlations between each quality signal and
per-example loss, a filtered re-evaluation curve and a logistic
`error ~ quality (+ truncation)` fit.

```bash
python scripts/link_translation_quality_to_detection.py --config configs/translation_quality_nllb.yaml --validate-only
python scripts/link_translation_quality_to_detection.py --config configs/translation_quality_nllb.yaml
```

Output: `linked_examples.parquet`, `stratified_metrics.csv`,
`filtered_curve.csv`, `link_summary.json`, `manifest.json`.

## Local (CPU) vs cluster (GPU)

- **Local:** Stage 0 in full, heuristics and truncation, reduced COMETKiwi/NLI
  smokes (`--limit`), and Stage 2 on a sample. The smoke config
  (`configs/translation_quality_smoke.yaml`) runs the whole pipeline on
  heuristics only.
- **Cluster (16 GB GPU):** set `scoring.metrics: [heuristics, cometkiwi,
  nli_consistency]`, `device: auto`, and run the full Stage 1 over all splits.
  No code change is required between CPU and GPU — only config/device/batch.
  COMETKiwi (`Unbabel/wmt22-cometkiwi-da`) is a gated repo: accept its license
  and provide an HF token.

## Cluster operational requirements

The code requires no changes between CPU and GPU; only the environment and
config/device/batch differ. Before the full run:

- Install a CUDA-matched torch first, then the project with the quality extra:
  `pip install -e ".[dev,docs,quality]"` (`unbabel-comet` resolves against the
  pinned `transformers`/`torch`; adjust the resolution if the cluster requires
  it).
- `Unbabel/wmt22-cometkiwi-da` is a gated repository: accept its license on the
  Hub and export `HF_TOKEN` so `snapshot_download(revision=...)` can fetch the
  pinned revision.
- Validate the real COMETKiwi loading path on the first cluster run
  (`snapshot_download` + `load_from_checkpoint`); it is exercised locally only
  through mocks because `unbabel-comet` is not installed on the CPU machine.
- This version has **no cache/resumability** for the heavy metrics: an
  interrupted COMETKiwi/NLI run restarts from scratch.

Full Stage 1 + Stage 2 per backend (after `--validate-only`):

```bash
python scripts/score_translation_quality.py --config configs/translation_quality_nllb.yaml --validate-only
python scripts/score_translation_quality.py --config configs/translation_quality_nllb.yaml
python scripts/link_translation_quality_to_detection.py --config configs/translation_quality_nllb.yaml --validate-only
python scripts/link_translation_quality_to_detection.py --config configs/translation_quality_nllb.yaml
# same four commands for configs/translation_quality_madlad.yaml
```

For PublicHearingBR, Stage 1 is ready with the same commands and the two
direction-specific configs. The configs intentionally have no `link` section:
the repository contains no PublicHearingBR PT→EN detector predictions with the
required `example_id`, `source_id`, `label`, `score` and frozen confirmatory
protocol contract. Existing PublicHearingBR CV outputs and RAGTruth transfer
runs are not silently promoted to that Stage 2 role.

## Confirmatory signatures

| Backend | Protocol signature |
|---|---|
| NLLB | `70bb1cce59b8c824` |
| MADLAD | `40828d496dd1d7ab` |

The configs point at the existing runs under `runs/`; no datasets or parquets
are copied or duplicated.
