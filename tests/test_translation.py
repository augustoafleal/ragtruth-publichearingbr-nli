from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ragtruth_transfer.ragtruth_parquet import load_ragtruth_parquet, split_ragtruth_parquet
from ragtruth_transfer.translation import TranslationConfig, TranslationSettings, translate_ragtruth


class FakeTranslator:
    def __init__(self, prefix: str = "PT:", incomplete: bool = False) -> None:
        self.prefix = prefix
        self.incomplete = incomplete
        self.calls: list[list[str]] = []

    def translate_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        result = [f"{self.prefix}{text}" for text in texts]
        return result[:-1] if self.incomplete else result


def _config(tmp_path: Path, backend: str = "nllb", *, max_rows: int | None = None) -> TranslationConfig:
    settings = TranslationSettings(
        translator=backend,
        model_name=f"{backend}-fake",
        source_language="eng_Latn" if backend == "nllb" else "en",
        target_language="por_Latn" if backend == "nllb" else "pt",
        batch_size=2,
        device="cpu",
    )
    return TranslationConfig(
        translation=settings,
        input_path=tmp_path / "source" / "dataset.parquet",
        output_dir=tmp_path / f"output-{backend}",
        manifest_path=tmp_path / "source" / "manifest.json",
        expected_source_signature="source",
        expected_source_schema="ragtruth-qa-training-view-deduplicated-v1",
        max_examples_per_split=max_rows,
    )


def _write_source(config: TranslationConfig, *, rows: int = 2) -> pd.DataFrame:
    config.input_path.parent.mkdir()
    values = []
    for index in range(rows):
        values.append({
            "example_id": f"e{2 - index}", "source_id": f"s{index}", "response_id": f"r{index}",
            "split": "test" if index == rows - 1 else "train", "claim": f"claim {index}", "label": int(index % 2 == 0),
            "claim_index": index, "claim_start": index * 10, "claim_end": index * 10 + 4,
            "evidence_mask": [True, True, True, False], "chunk_1": "shared evidence", "chunk_2": f"second {index}", "chunk_3": f"third {index}", "chunk_4": "",
            "chunk_1_sha256": f"hash-{index}", "chunk_1_token_start": 0, "chunk_1_token_end": 4,
            "retrieval_signature": "retrieval", "chunking_signature": "chunking", "metadata": f"keep-{index}",
        })
    frame = pd.DataFrame(values)
    frame.to_parquet(config.input_path, index=False)
    digest = hashlib.sha256(config.input_path.read_bytes()).hexdigest()
    assert config.manifest_path is not None
    config.manifest_path.write_text(json.dumps({"schema_version": "ragtruth-qa-training-view-deduplicated-v1", "signature": "source", "artifacts": {"dataset.parquet": digest}}), encoding="utf-8")
    return frame


def test_existing_backend_configs_use_the_canonical_parquet() -> None:
    configs = [TranslationConfig.from_yaml(Path(name)) for name in (
        "configs/ragtruth_translate_nllb.yaml", "configs/ragtruth_translate_nllb_smoke.yaml",
        "configs/ragtruth_translate_madlad.yaml", "configs/ragtruth_translate_madlad_smoke.yaml",
    )]
    assert all(config.input_path.name == "dataset.parquet" for config in configs)
    assert configs[0].translation.model_name == "facebook/nllb-200-distilled-600M"
    assert configs[0].translation.batch_size == 16 and configs[0].translation.num_beams == 4
    assert configs[2].translation.model_name == "google/madlad400-3b-mt"
    assert configs[2].translation.batch_size == 16 and configs[2].translation.num_beams == 4
    assert configs[3].translation.batch_size == 2 and configs[3].translation.num_beams == 4


@pytest.mark.parametrize("backend", ["nllb", "madlad"])
def test_shared_parquet_pipeline_preserves_structure_and_runs_qa(tmp_path: Path, backend: str) -> None:
    config = _config(tmp_path, backend)
    source = _write_source(config)
    fake = FakeTranslator(prefix=f"{backend}:")
    manifest = translate_ragtruth(config, translator=fake)
    output = pd.read_parquet(config.output_dir / "dataset.parquet")
    assert output.columns.tolist() == source.columns.tolist()
    assert output["example_id"].tolist() == source["example_id"].tolist()
    assert output["claim"].tolist() == [f"{backend}:claim 0", f"{backend}:claim 1"]
    assert output.loc[0, "chunk_1"] == f"{backend}:shared evidence"
    assert output.loc[0, "chunk_2"] == f"{backend}:second 0"
    assert output.loc[0, "chunk_3"] == f"{backend}:third 0"
    assert output.loc[0, "chunk_4"] == ""
    for column in ("source_id", "response_id", "label", "split", "claim_start", "chunk_1_sha256", "retrieval_signature", "metadata"):
        assert output[column].tolist() == source[column].tolist()
    assert [list(value) for value in output["evidence_mask"]] == [list(value) for value in source["evidence_mask"]]
    assert manifest["generation"]["batch_size"] == 2
    assert "requested_model_revision" in manifest and "resolved_model_revision" in manifest
    assert manifest["qa"]["rows_removed"] == 0
    assert manifest["qa"]["pairs"] == 8
    assert (config.output_dir / "translation_qa.json").is_file()
    assert (config.output_dir / "translation_qa_flags.parquet").is_file()
    assert sum(len(call) for call in fake.calls) == len({"claim 0", "claim 1", "shared evidence", "second 0", "second 1", "third 0", "third 1"})


@pytest.mark.parametrize("backend", ["nllb", "madlad"])
def test_missing_translation_is_an_error_for_every_backend(tmp_path: Path, backend: str) -> None:
    config = _config(tmp_path, backend)
    _write_source(config)
    with pytest.raises(RuntimeError, match="incompleto"):
        translate_ragtruth(config, translator=FakeTranslator(incomplete=True))
    assert not (config.output_dir / "dataset.parquet").exists()


def test_cache_isolated_by_backend_and_incompatible_output_fails(tmp_path: Path) -> None:
    nllb = _config(tmp_path, "nllb")
    _write_source(nllb)
    shared_cache = tmp_path / "cache.sqlite3"
    nllb = replace(nllb, cache_path=shared_cache)
    translate_ragtruth(nllb, translator=FakeTranslator())
    madlad = replace(_config(tmp_path, "madlad"), cache_path=shared_cache)
    with pytest.raises(ValueError, match="Cache de tradução incompatível"):
        translate_ragtruth(madlad, translator=FakeTranslator())
    with pytest.raises(FileExistsError, match="já existe"):
        translate_ragtruth(nllb, translator=FakeTranslator())


def test_split_assignments_are_validated_without_reordering_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(config, rows=12)
    source.loc[10:, "split"] = "test"
    source.to_parquet(config.input_path, index=False)
    digest = hashlib.sha256(config.input_path.read_bytes()).hexdigest()
    assert config.manifest_path is not None
    config.manifest_path.write_text(json.dumps({"schema_version": "ragtruth-qa-training-view-deduplicated-v1", "signature": "source", "artifacts": {"dataset.parquet": digest}}), encoding="utf-8")
    rows, metadata = load_ragtruth_parquet(config.input_path, manifest_path=config.manifest_path, expected_signature="source")
    reference = split_ragtruth_parquet(rows, metadata, split_seed=42, validation_fraction=.15, max_test_sources=None)
    reference_path = tmp_path / "assignments.parquet"
    reference.assignments.to_parquet(reference_path, index=False)
    manifest = translate_ragtruth(replace(config, reference_split_assignments=reference_path), translator=FakeTranslator())
    assert manifest["split_assignment_validation"]["checked"] is True
    assert pd.read_parquet(config.output_dir / "dataset.parquet")["example_id"].tolist() == source["example_id"].tolist()


def test_resume_completes_an_integrity_checked_validating_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = _write_source(config, rows=12)
    source.loc[10:, "split"] = "test"
    source.to_parquet(config.input_path, index=False)
    digest = hashlib.sha256(config.input_path.read_bytes()).hexdigest()
    assert config.manifest_path is not None
    config.manifest_path.write_text(json.dumps({"schema_version": "ragtruth-qa-training-view-deduplicated-v1", "signature": "source", "artifacts": {"dataset.parquet": digest}}), encoding="utf-8")
    rows, metadata = load_ragtruth_parquet(config.input_path, manifest_path=config.manifest_path, expected_signature="source")
    reference = split_ragtruth_parquet(rows, metadata, split_seed=42, validation_fraction=.15, max_test_sources=None)
    reference_path = tmp_path / "assignments.parquet"
    reference.assignments.to_parquet(reference_path, index=False)
    config = replace(config, reference_split_assignments=reference_path)
    translate_ragtruth(config, translator=FakeTranslator())

    manifest_path = config.output_dir / "manifest.json"
    pending = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending["status"] = "validating"
    manifest_path.write_text(json.dumps(pending), encoding="utf-8")

    recovery_translator = FakeTranslator()
    resumed = translate_ragtruth(config, translator=recovery_translator, resume=True)

    assert resumed["status"] == "completed"
    assert resumed["split_assignment_validation"]["checked"] is True
    assert recovery_translator.calls == []
