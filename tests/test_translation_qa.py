from __future__ import annotations

import json
from pathlib import Path

from ragtruth_transfer.translation_qa import (
    classify_translation_pair,
    compute_length_metrics,
    compute_repetition_metrics,
)
from scripts.filter_ragtruth_translation_failures import filter_dataset


def test_length_metrics_cover_normal_low_high_and_empty() -> None:
    assert compute_length_metrics("short source", "fonte curta")["low_ratio"] is False
    assert compute_length_metrics("x" * 100, "curto")["low_ratio"] is True
    assert compute_length_metrics("short", "x" * 20)["high_ratio"] is True
    empty = compute_length_metrics("source", "")
    assert empty["empty_translation"] is True
    assert empty["length_ratio"] == 0.0
    assert compute_length_metrics("", "target")["length_ratio"] is None


def test_repetition_distinguishes_inherited_and_target_added_repetition() -> None:
    inherited = compute_repetition_metrics("very very very very", "muito muito muito muito")
    assert inherited["source_has_repetition"] is True
    assert inherited["translation_has_repetition"] is True
    assert inherited["repetition_excess"] == 0
    assert inherited["high_confidence_repetition"] is False

    added = compute_repetition_metrics("the proposal was approved", "a a a a a proposta")
    assert added["high_confidence_repetition"] is True
    assert added["repetition_excess"] >= 3

    clean = compute_repetition_metrics("the proposal was approved", "a proposta foi aprovada")
    assert clean["translation_has_repetition"] is False
    assert clean["high_confidence_repetition"] is False


def test_candidate_rule_is_exact_union() -> None:
    low = classify_translation_pair("x" * 100, "short")
    high = classify_translation_pair("short", "x" * 20)
    repeated = classify_translation_pair("the proposal", "a a a a a proposta")
    clean = classify_translation_pair("the proposal", "a proposta")
    assert low["exclusion_candidate"] is True and low["low_ratio"] is True
    assert high["exclusion_candidate"] is True and high["high_ratio"] is True
    assert repeated["exclusion_candidate"] is True and repeated["high_confidence_repetition"] is True
    assert clean["exclusion_candidate"] is False


def _row(example_id: str, label: bool, claim: str, translated_claim: str, evidence: list[str], translated_evidence: list[str], mask: list[bool]) -> tuple[dict, dict]:
    source = {
        "example_id": example_id,
        "source_id": f"source-{example_id}",
        "label": label,
        "claim": claim,
        "evidence": evidence,
        "evidence_mask": mask,
        "metadata": {"keep": example_id},
    }
    translated = {
        **source,
        "claim": translated_claim,
        "evidence": translated_evidence,
    }
    return source, translated


def _write_dataset(root: Path) -> tuple[Path, Path, dict[str, list[dict]], dict[str, list[dict]]]:
    source_dir = root / "source"
    translated_dir = root / "translated"
    source_dir.mkdir()
    translated_dir.mkdir()
    rows = {
        "train": [
            _row("clean", False, "The cat is here", "O gato está aqui", ["Cats are animals"], ["Gatos são animais"], [True]),
            _row("bad-claim", True, "x" * 100, "curto", ["Valid evidence"], ["Evidência válida"], [True]),
            _row("bad-evidence", False, "The claim is clean", "A alegação é limpa", ["e" * 100], ["curto"], [True]),
            _row("masked-bad", True, "The claim is clean", "A alegação é limpa", ["e" * 100], ["e" * 100], [False]),
        ],
        "validation": [
            _row("bad-repeat", True, "The claim is clean", "A alegação é limpa", ["The answer is clear"], ["a a a a a resposta"], [True]),
            _row("bad-both", False, "z" * 100, "curto", ["The answer is clear"], ["a a a a a resposta"], [True]),
        ],
        "test": [
            _row("bad-high", False, "hi", "esta tradução é muito maior", ["The answer is clear"], ["A resposta é clara"], [True]),
        ],
    }
    source_rows = {split: [pair[0] for pair in values] for split, values in rows.items()}
    translated_rows = {split: [pair[1] for pair in values] for split, values in rows.items()}
    for directory, values in ((source_dir, source_rows), (translated_dir, translated_rows)):
        for split, split_rows in values.items():
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
                encoding="utf-8",
            )
    return source_dir, translated_dir, source_rows, translated_rows


def test_filtering_masks_valid_evidence_only_preserves_order_and_is_deterministic(tmp_path: Path) -> None:
    source_dir, translated_dir, source_rows, translated_rows = _write_dataset(tmp_path)
    qa_config = {
        "min_length_ratio": 0.40,
        "max_length_ratio": 2.50,
        "repetition_min_run": 4,
        "expected_candidate_translation_pairs": 5,
    }
    first_output = tmp_path / "filtered"
    first_manifest = filter_dataset(
        root=tmp_path,
        source_dir=source_dir,
        translated_dir=translated_dir,
        output_dir=first_output,
        qa_config=qa_config,
    )
    first_bytes = {path.name: path.read_bytes() for path in first_output.glob("*.jsonl")}
    first_manifest_bytes = (first_output / "manifest.json").read_bytes()

    assert first_manifest["counts"]["candidate_translation_pairs"] == 5
    assert first_manifest["counts"]["excluded_examples"] == 5
    assert first_manifest["counts"]["remaining_examples"] == 2
    assert first_manifest["counts"]["per_split"]["train"]["excluded_examples"] == 2
    assert first_manifest["counts"]["per_split"]["validation"]["excluded_examples"] == 2
    assert first_manifest["counts"]["per_split"]["test"]["excluded_examples"] == 1

    remaining = {
        split: [json.loads(line) for line in (first_output / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "validation", "test")
    }
    assert [row["example_id"] for row in remaining["train"]] == ["clean", "masked-bad"]
    assert [row["example_id"] for row in remaining["validation"]] == []
    assert [row["example_id"] for row in remaining["test"]] == []
    assert remaining["train"] == [translated_rows["train"][0], translated_rows["train"][3]]
    assert first_manifest["counts"]["per_label"]["excluded"] == {"0": 3, "1": 2}

    second_output = first_output
    second_manifest = filter_dataset(
        root=tmp_path,
        source_dir=source_dir,
        translated_dir=translated_dir,
        output_dir=second_output,
        qa_config=qa_config,
        force=True,
    )
    assert second_manifest["run_signature"] == first_manifest["run_signature"]
    assert second_manifest["counts"] == first_manifest["counts"]
    assert {path.name: path.read_bytes() for path in second_output.glob("*.jsonl")} == first_bytes
    assert (second_output / "manifest.json").read_bytes() == first_manifest_bytes
