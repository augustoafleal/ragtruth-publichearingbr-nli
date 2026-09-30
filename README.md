# RAGTruth and PublicHearingBR-NLI

Reproducible data preparation, detector experiments, zero-shot transfer, and
paper reporting for the RAGTruth and PublicHearingBR-NLI study.

The repository includes:

- PublicHearingBR supervised and RAGTruth confirmatory experiments.
- NLI and threshold-transfer baselines.
- Pooling ablations and paired bootstrap analyses.
- Translation-quality analyses.
- Offline LLM tokenization and current-equivalent USD cost analysis.
- NLLB-only confirmatory paper figures, with the historical five-condition
  reporting mode preserved.

Frozen manifests, resolved configurations, predictions, metrics, and reports in
`runs/` and `results/` are the source of truth for completed runs.

## Results

Under zero-shot transfer from English RAGTruth to Portuguese PublicHearingBR, the Set Transformer achieved **0.6035 AUPRC**, compared with **0.3375** for the off-the-shelf multilingual NLI baseline: a **78.8% relative improvement** without target-domain supervision.

See the [documentation](https://augustoafleal.github.io/ragtruth-publichearingbr-nli/) for the main experimental results and comparisons.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

## Documentation

Build or serve the documentation site:

```bash
mkdocs serve
mkdocs build --strict
```

Start with the [documentation home](docs/index.md), then use:

- [Preparation](docs/preparation/index.md) for the environment and datasets.
- [Experiment inventory](docs/experiments/experiment-inventory.md) for the
  canonical experiment families.
- [Pooling ablation](docs/experiments/pooling-ablation.md) for detector
  variants.
- [Confirmatory paper figures](docs/experiments/confirmatory-paper-figures.md)
  for the NLLB-only presentation mode.
- [Offline LLM cost analysis](docs/experiments/llm-cost-analysis.md) for
  tokenization and USD pricing.
- [Reporting CI](docs/experiments/paper-reporting-ci.md) for frozen-artifact
  validation.

## Quick start

Prepare the RAGTruth data view:

```bash
python scripts/download_ragtruth.py --output-dir data/raw/ragtruth
python scripts/prepare_ragtruth_top4.py \
  --config configs/ragtruth_qa_top4_label_independent.yaml \
  --device cuda \
  --batch-size 64 \
  --resume
python scripts/build_ragtruth_training_view.py \
  --input-run-dir results/ragtruth_qa_top4_label_independent/ragtruth_qa_top4_label_independent_embeddings/<signature> \
  --output-root results/ragtruth_qa_training_view
```

Run the canonical detector experiment:

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

Use the experiment pages for translation variants, pooling configurations,
baselines, threshold transfer, bootstrap analyses, and reporting commands.

## Local validation

```bash
python -m compileall -q src scripts
pytest -q -m "not frozen_artifacts"
mkdocs build --strict
```

Tests that require unversioned frozen artifacts are documented in
[Reporting CI](docs/experiments/paper-reporting-ci.md).

## Citation

If you use this work, code, or experimental results, please cite:
```bibtex
@misc{leal2026crosslingual,
  title  = {Cross-Lingual Transfer for Evidence-Based Hallucination Detection in PublicHearingBR},
  author = {Leal, Augusto Antônio Fontanive and
            de Souza, Arturo and
            de Brum, Antônio Araújo and
            Laner, João Augusto Tonial},
  year   = {2026}
}
```
