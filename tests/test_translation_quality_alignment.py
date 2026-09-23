from __future__ import annotations

import pandas as pd
import pytest

from ragtruth_transfer.ragtruth_top4_embeddings import sha256_text
from ragtruth_transfer.translation_quality.alignment import (
    AlignmentIntegrityError,
    align_translation_quality,
)
from ragtruth_transfer.translation_quality.config import AlignmentConfig

EN_CLAIM = "the cat sat on the mat"
PT_CLAIM = "o gato sentou no tapete"
EN_CHUNKS = ["a cat was sitting", "the animal rested", "", ""]
PT_CHUNKS = ["um gato estava sentado", "o animal descansou", "", ""]
MASK = [True, True, False, False]
PROVENANCE = [
    (0, 0, 0, 4),
    (0, 1, 3, 7),
    (-1, -1, 0, 0),
    (-1, -1, 0, 0),
]


def _base_row(example_id: str, *, source_id: str = "src-1", label: bool = True) -> dict:
    row = {
        "example_id": example_id,
        "source_id": source_id,
        "response_id": "resp-1",
        "split": "test",
        "task": "QA",
        "label": label,
        "evidence_mask": list(MASK),
        "chunking_signature": "sig",
        "tokenizer_revision": "rev",
    }
    return row


def _make_frames(extra_pt: dict | None = None):
    en = _base_row("ex-1")
    pt = _base_row("ex-1")
    for prefix, claim, chunks in (("en", EN_CLAIM, EN_CHUNKS), ("pt", PT_CLAIM, PT_CHUNKS)):
        target = en if prefix == "en" else pt
        target["claim"] = claim
        for slot, (text, (si, wi, ts, te)) in enumerate(zip(chunks, PROVENANCE), start=1):
            target[f"chunk_{slot}"] = text
            target[f"chunk_{slot}_source_index"] = si
            target[f"chunk_{slot}_window_index"] = wi
            target[f"chunk_{slot}_token_start"] = ts
            target[f"chunk_{slot}_token_end"] = te
            target[f"chunk_{slot}_sha256"] = sha256_text(EN_CHUNKS[slot - 1]) if MASK[slot - 1] else ""
    pt_rows = [pt]
    if extra_pt is not None:
        pt_rows.append(extra_pt)
    return pd.DataFrame([en]), pd.DataFrame(pt_rows)


def _config(tmp_path, en: pd.DataFrame, pt: pd.DataFrame) -> AlignmentConfig:
    en_path = tmp_path / "en.parquet"
    pt_path = tmp_path / "pt.parquet"
    en.to_parquet(en_path, index=False)
    pt.to_parquet(pt_path, index=False)
    return AlignmentConfig(
        backend="test",
        en_parquet=en_path,
        pt_parquet=pt_path,
        output_dir=tmp_path / "out",
        expected_rows=len(pt),
        expected_chunking_signature="sig",
        expected_tokenizer_revision="rev",
    )


def test_alignment_succeeds_and_writes_artifacts(tmp_path):
    en, pt = _make_frames()
    config = _config(tmp_path, en, pt)

    result = align_translation_quality(config)

    assert result.counts.gate_ok
    assert result.counts.total_valid_chunks == 2
    assert result.counts.en_chunk_sha256_verified == 2
    assert result.counts.en_chunk_sha256_mismatches == 0
    record = result.frame.iloc[0]
    assert record["claim_en"] == EN_CLAIM
    assert record["claim_pt"] == PT_CLAIM
    assert bool(record["chunk_1_valid"]) is True
    assert record["chunk_1_en"] == EN_CHUNKS[0]
    assert record["chunk_1_pt"] == PT_CHUNKS[0]
    assert bool(record["chunk_3_valid"]) is False
    assert (config.run_dir / "aligned.parquet").is_file()
    assert (config.run_dir / "manifest.json").is_file()


def test_alignment_validate_only_does_not_write(tmp_path):
    en, pt = _make_frames()
    config = _config(tmp_path, en, pt)
    align_translation_quality(config, validate_only=True)
    assert not (config.run_dir / "aligned.parquet").exists()


def test_alignment_fails_on_missing_in_en(tmp_path):
    extra = _base_row("ex-2")
    extra["claim"] = "outra frase"
    for slot in range(1, 5):
        extra[f"chunk_{slot}"] = ""
        extra[f"chunk_{slot}_source_index"] = -1
        extra[f"chunk_{slot}_window_index"] = -1
        extra[f"chunk_{slot}_token_start"] = 0
        extra[f"chunk_{slot}_token_end"] = 0
        extra[f"chunk_{slot}_sha256"] = ""
    en, pt = _make_frames(extra_pt=extra)
    config = _config(tmp_path, en, pt)
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(config, validate_only=True)
    assert error.value.counts.missing_in_en == 1


def test_alignment_fails_on_source_id_mismatch(tmp_path):
    en, pt = _make_frames()
    pt.loc[0, "source_id"] = "other-src"
    config = _config(tmp_path, en, pt)
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(config, validate_only=True)
    assert error.value.counts.source_id_mismatches == 1


def test_alignment_fails_on_label_mismatch(tmp_path):
    en, pt = _make_frames()
    pt.loc[0, "label"] = False
    config = _config(tmp_path, en, pt)
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(config, validate_only=True)
    assert error.value.counts.label_mismatches == 1


def test_alignment_fails_on_provenance_mismatch(tmp_path):
    en, pt = _make_frames()
    pt.loc[0, "chunk_1_token_start"] = 99
    config = _config(tmp_path, en, pt)
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(config, validate_only=True)
    assert error.value.counts.provenance_mismatches >= 1


def test_alignment_fails_on_en_chunk_sha_mismatch(tmp_path):
    en, pt = _make_frames()
    en.loc[0, "chunk_1"] = "tampered english chunk"
    config = _config(tmp_path, en, pt)
    with pytest.raises(AlignmentIntegrityError) as error:
        align_translation_quality(config, validate_only=True)
    assert error.value.counts.en_chunk_sha256_mismatches == 1


def test_alignment_signature_is_machine_independent(tmp_path):
    en, pt = _make_frames()
    config = _config(tmp_path, en, pt)
    relocated = AlignmentConfig(
        backend=config.backend,
        en_parquet=tmp_path / "relocated" / "en.parquet",
        pt_parquet=tmp_path / "relocated" / "pt.parquet",
        output_dir=tmp_path / "elsewhere",
        expected_rows=config.expected_rows,
        expected_chunking_signature=config.expected_chunking_signature,
        expected_tokenizer_revision=config.expected_tokenizer_revision,
    )
    assert relocated.signature == config.signature
