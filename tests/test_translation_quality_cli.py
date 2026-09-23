from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ragtruth_transfer.ragtruth_top4_embeddings import sha256_text

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = "abcdef0123456789"


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _row(example_id: str, prefix: str) -> dict:
    claim_en = "the cat sat on the mat"
    claim_pt = "o gato sentou no tapete"
    chunk_en = "a cat was sitting"
    chunk_pt = "um gato estava sentado"
    row = {
        "example_id": example_id,
        "source_id": "src-1",
        "response_id": "resp-1",
        "split": "test",
        "label": True,
        "evidence_mask": [True, False, False, False],
        "claim": claim_pt if prefix == "pt" else claim_en,
        "chunking_signature": "sig",
        "tokenizer_revision": "rev",
    }
    for slot in range(1, 5):
        valid = slot == 1
        row[f"chunk_{slot}"] = (chunk_pt if prefix == "pt" else chunk_en) if valid else ""
        row[f"chunk_{slot}_source_index"] = 0 if valid else -1
        row[f"chunk_{slot}_window_index"] = 0 if valid else -1
        row[f"chunk_{slot}_token_start"] = 0 if valid else 0
        row[f"chunk_{slot}_token_end"] = 4 if valid else 0
        row[f"chunk_{slot}_sha256"] = sha256_text(chunk_en) if valid else ""
    return row


def _write_fixture(tmp_path: Path) -> Path:
    pd.DataFrame([_row("ex-1", "en")]).to_parquet(tmp_path / "en.parquet", index=False)
    pd.DataFrame([_row("ex-1", "pt")]).to_parquet(tmp_path / "pt.parquet", index=False)

    preds_dir = tmp_path / "runs" / "ragtruth_pt_smoke_confirmatory" / SIGNATURE / "seed_0"
    (preds_dir / "ragtruth_test").mkdir(parents=True)
    pd.DataFrame(
        [{"example_id": "ex-1", "source_id": "src-1", "label": True, "score": 0.9}]
    ).to_parquet(preds_dir / "ragtruth_test" / "predictions.parquet", index=False)
    (preds_dir / "thresholds.json").write_text(
        json.dumps({"f1": {"threshold": 0.5, "rule": "protocol"}}), encoding="utf-8"
    )

    config_path = tmp_path / "translation_quality_smoke.yaml"
    config_path.write_text(
        f"""
backend: smoke
alignment:
  en_parquet: en.parquet
  pt_parquet: pt.parquet
  output_dir: out/alignment
  expected_rows: 1
  expected_chunking_signature: sig
  expected_tokenizer_revision: rev
scoring:
  output_dir: out/scoring
  metrics: [heuristics]
  device: cpu
  sample_split: test
link:
  output_dir: out/link
  predictions_parquet: runs/ragtruth_pt_smoke_confirmatory/{SIGNATURE}/seed_0/ragtruth_test/predictions.parquet
  protocol_thresholds_json: runs/ragtruth_pt_smoke_confirmatory/{SIGNATURE}/seed_0/thresholds.json
  threshold_criterion: f1
  expected_protocol_signature: {SIGNATURE}
  quality_signal: any_exclusion_candidate
  quality_direction: lower_is_better
  num_quality_buckets: 2
  bootstrap_samples: 20
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_pipeline_validate_only_and_run(tmp_path):
    config_path = _write_fixture(tmp_path)

    align_validate = _run("align_translation_quality.py", "--config", str(config_path), "--validate-only")
    assert align_validate.returncode == 0, align_validate.stderr
    assert '"gate_ok": true' in align_validate.stdout
    assert '"validate_only": true' in align_validate.stdout
    assert '"artifacts_written": false' in align_validate.stdout
    assert '"aligned_parquet":' in align_validate.stdout

    align = _run("align_translation_quality.py", "--config", str(config_path))
    assert align.returncode == 0, align.stderr
    assert '"artifacts_written": true' in align.stdout

    score_validate = _run("score_translation_quality.py", "--config", str(config_path), "--validate-only")
    assert score_validate.returncode == 0, score_validate.stderr
    assert "validate_only" in score_validate.stdout

    score = _run("score_translation_quality.py", "--config", str(config_path))
    assert score.returncode == 0, score.stderr

    link_validate = _run(
        "link_translation_quality_to_detection.py", "--config", str(config_path), "--validate-only"
    )
    assert link_validate.returncode == 0, link_validate.stderr
    assert '"threshold": 0.5' in link_validate.stdout

    link = _run("link_translation_quality_to_detection.py", "--config", str(config_path))
    assert link.returncode == 0, link.stderr


def test_cli_align_fails_on_integrity_error(tmp_path):
    config_path = _write_fixture(tmp_path)
    pt = pd.read_parquet(tmp_path / "pt.parquet")
    pt.loc[0, "source_id"] = "different-source"
    pt.to_parquet(tmp_path / "pt.parquet", index=False)

    result = _run("align_translation_quality.py", "--config", str(config_path), "--validate-only")
    assert result.returncode == 1
    assert "FALHOU" in result.stderr
