#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ragtruth_transfer.io_utils import read_jsonl, sha256_file
from ragtruth_transfer.translation_qa import (
    classify_translation_pair,
    stable_pair_id,
    structural_mismatch_counts,
)

SPLITS = ("train", "validation", "test")
EXPECTED_CANDIDATE_TRANSLATION_PAIRS = 216
FILTER_MANIFEST_SCHEMA = "ragtruth-translated-filtered-v1"


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Repository root was not found.")


def repository_path(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative.")
    return root / path


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter("1" if bool(row["label"]) else "0" for row in rows)
    return {"0": int(counts.get("0", 0)), "1": int(counts.get("1", 0))}


def _split_counts(original: list[dict[str, Any]], excluded: list[dict[str, Any]], remaining: list[dict[str, Any]]) -> dict[str, Any]:
    original_labels = _label_counts(original)
    excluded_labels = _label_counts(excluded)
    remaining_labels = _label_counts(remaining)
    return {
        "original_examples": len(original),
        "excluded_examples": len(excluded),
        "remaining_examples": len(remaining),
        "excluded_percentage": (100.0 * len(excluded) / len(original)) if original else 0.0,
        "by_label": {
            "0": {
                "original": original_labels["0"],
                "excluded": excluded_labels["0"],
                "remaining": remaining_labels["0"],
            },
            "1": {
                "original": original_labels["1"],
                "excluded": excluded_labels["1"],
                "remaining": remaining_labels["1"],
            },
        },
    }


def _validate_alignment(split: str, source_rows: list[dict[str, Any]], translated_rows: list[dict[str, Any]]) -> None:
    if len(source_rows) != len(translated_rows):
        raise ValueError(f"{split}: source/PT row count mismatch: {len(source_rows)} != {len(translated_rows)}")
    for index, (source, translated) in enumerate(zip(source_rows, translated_rows)):
        mismatches = structural_mismatch_counts(source, translated)
        failures = {name: value for name, value in mismatches.items() if value}
        if failures:
            raise ValueError(f"{split}[{index}] structural mismatch: {failures}")


def _pair_is_candidate(
    source_text: str,
    translated_text: str,
    cache: dict[tuple[str, str], dict[str, Any]],
    qa_config: dict[str, Any],
) -> dict[str, Any]:
    key = (source_text, translated_text)
    if key not in cache:
        cache[key] = classify_translation_pair(
            source_text,
            translated_text,
            min_length_ratio=float(qa_config.get("min_length_ratio", 0.40)),
            max_length_ratio=float(qa_config.get("max_length_ratio", 2.50)),
            repetition_min_run=int(qa_config.get("repetition_min_run", 4)),
        )
    return cache[key]


def filter_dataset(
    *,
    root: Path,
    source_dir: Path,
    translated_dir: Path,
    output_dir: Path,
    qa_config: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    translated_dir = translated_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir in {source_dir, translated_dir}:
        raise ValueError("Filtered output must differ from both input datasets.")
    if output_dir.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to replace it: {output_dir}")

    source_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    translated_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    input_hashes = {"source": {}, "translated": {}}
    for split in SPLITS:
        source_path = source_dir / f"{split}.jsonl"
        translated_path = translated_dir / f"{split}.jsonl"
        source_rows_by_split[split] = read_jsonl(source_path)
        translated_rows_by_split[split] = read_jsonl(translated_path)
        _validate_alignment(split, source_rows_by_split[split], translated_rows_by_split[split])
        input_hashes["source"][split] = sha256_file(source_path)
        input_hashes["translated"][split] = sha256_file(translated_path)

    pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_pair_ids: set[str] = set()
    candidate_pair_keys: set[tuple[str, str]] = set()
    excluded_translation_occurrences = 0
    output_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    excluded_rows_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in SPLITS:
        output_rows: list[dict[str, Any]] = []
        excluded_rows: list[dict[str, Any]] = []
        for source, translated in zip(source_rows_by_split[split], translated_rows_by_split[split]):
            relevant_pairs = [(source["claim"], translated["claim"])]
            relevant_pairs.extend(
                (source["evidence"][index], translated["evidence"][index])
                for index, is_valid in enumerate(source["evidence_mask"])
                if is_valid is True
            )
            example_excluded = False
            for source_text, translated_text in relevant_pairs:
                metrics = _pair_is_candidate(source_text, translated_text, pair_cache, qa_config)
                if metrics["exclusion_candidate"]:
                    example_excluded = True
                    excluded_translation_occurrences += 1
                    identity = (metrics["source_normalized"], metrics["translated_normalized"])
                    candidate_pair_keys.add(identity)
                    candidate_pair_ids.add(stable_pair_id(*identity))
            if example_excluded:
                excluded_rows.append(translated)
            else:
                output_rows.append(translated)
        output_rows_by_split[split] = output_rows
        excluded_rows_by_split[split] = excluded_rows

    expected_pairs = int(qa_config.get("expected_candidate_translation_pairs", EXPECTED_CANDIDATE_TRANSLATION_PAIRS))
    if len(candidate_pair_keys) != expected_pairs:
        raise AssertionError(
            "Candidate translation-pair regression mismatch: "
            f"expected {expected_pairs}, observed {len(candidate_pair_keys)}."
        )

    config_payload = {
        "source_dir": str(source_dir.relative_to(root.resolve())),
        "translated_dir": str(translated_dir.relative_to(root.resolve())),
        "output_dir": str(output_dir.relative_to(root.resolve())),
        "qa": {
            "min_length_ratio": float(qa_config.get("min_length_ratio", 0.40)),
            "max_length_ratio": float(qa_config.get("max_length_ratio", 2.50)),
            "repetition_min_run": int(qa_config.get("repetition_min_run", 4)),
        },
    }
    config_signature = _sha256_payload(config_payload)
    run_signature = _sha256_payload({"inputs": input_hashes, "config_signature": config_signature})

    split_counts = {
        split: _split_counts(
            source_rows_by_split[split],
            excluded_rows_by_split[split],
            output_rows_by_split[split],
        )
        for split in SPLITS
    }
    all_original = [row for rows in source_rows_by_split.values() for row in rows]
    all_excluded = [row for rows in excluded_rows_by_split.values() for row in rows]
    all_remaining = [row for rows in output_rows_by_split.values() for row in rows]
    manifest = {
        "schema_version": FILTER_MANIFEST_SCHEMA,
        "status": "completed",
        "run_signature": run_signature,
        "source_dataset": str(source_dir.relative_to(root.resolve())),
        "translated_dataset": str(translated_dir.relative_to(root.resolve())),
        "output_dataset": str(output_dir.relative_to(root.resolve())),
        "qa_config": config_payload["qa"],
        "exclusion_policy": "low_ratio OR high_ratio OR high_confidence_repetition",
        "counts": {
            "candidate_translation_pairs": len(candidate_pair_keys),
            "candidate_translation_pair_ids": sorted(candidate_pair_ids),
            "excluded_translation_occurrences": excluded_translation_occurrences,
            "excluded_unique_examples": len(all_excluded),
            "original_examples": len(all_original),
            "excluded_examples": len(all_excluded),
            "remaining_examples": len(all_remaining),
            "per_split": split_counts,
            "per_label": {
                "original": _label_counts(all_original),
                "excluded": _label_counts(all_excluded),
                "remaining": _label_counts(all_remaining),
            },
        },
        "inputs": input_hashes,
        "config_signature": config_signature,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for split in SPLITS:
            _write_jsonl(stage / f"{split}.jsonl", output_rows_by_split[split])
        manifest["output_hashes"] = {
            split: sha256_file(stage / f"{split}.jsonl") for split in SPLITS
        }
        manifest["dataset_sha256"] = _sha256_payload(manifest["output_hashes"])
        manifest["split_signature"] = _sha256_payload(
            {"splits": manifest["output_hashes"]}
        )[:16]
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            if not force:
                raise FileExistsError(f"Output appeared during filtering: {output_dir}")
            shutil.rmtree(output_dir)
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def validate_filtered_dataset(
    *,
    root: Path,
    source_dir: Path,
    translated_dir: Path,
    output_dir: Path,
    qa_config: dict[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    source_dir = source_dir.resolve()
    translated_dir = translated_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Filtered output/manifest not found: {output_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed filtered manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError("Filtered manifest must be a completed object.")
    if manifest.get("schema_version") != FILTER_MANIFEST_SCHEMA:
        raise ValueError("Filtered manifest schema is incompatible.")

    input_hashes = {"source": {}, "translated": {}}
    for kind, directory in (("source", source_dir), ("translated", translated_dir)):
        for split in SPLITS:
            file_path = directory / f"{split}.jsonl"
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing {kind} split: {file_path}")
            input_hashes[kind][split] = sha256_file(file_path)
    if manifest.get("inputs") != input_hashes:
        raise ValueError("Filtered manifest inputs do not match current datasets.")

    config_payload = {
        "source_dir": str(source_dir.relative_to(root)),
        "translated_dir": str(translated_dir.relative_to(root)),
        "output_dir": str(output_dir.relative_to(root)),
        "qa": {
            "min_length_ratio": float(qa_config.get("min_length_ratio", 0.40)),
            "max_length_ratio": float(qa_config.get("max_length_ratio", 2.50)),
            "repetition_min_run": int(qa_config.get("repetition_min_run", 4)),
        },
    }
    config_signature = _sha256_payload(config_payload)
    run_signature = _sha256_payload({"inputs": input_hashes, "config_signature": config_signature})
    if manifest.get("config_signature") != config_signature or manifest.get("run_signature") != run_signature:
        raise ValueError("Filtered manifest configuration/signature is incompatible.")

    output_hashes = manifest.get("output_hashes")
    if not isinstance(output_hashes, dict):
        raise ValueError("Filtered manifest does not contain output_hashes.")
    for split in SPLITS:
        file_path = output_dir / f"{split}.jsonl"
        if not file_path.is_file() or sha256_file(file_path) != str(output_hashes.get(split)):
            raise ValueError(f"Filtered split missing or hash mismatch: {split}")
        expected = manifest.get("counts", {}).get("per_split", {}).get(split, {}).get("remaining_examples")
        actual = len(read_jsonl(file_path))
        if expected is None or int(expected) != actual:
            raise ValueError(f"Filtered split count mismatch in {split}: {actual} != {expected}")
    expected_dataset_sha = _sha256_payload(output_hashes)
    expected_split_signature = _sha256_payload({"splits": output_hashes})[:16]
    if manifest.get("dataset_sha256") != expected_dataset_sha or manifest.get("split_signature") != expected_split_signature:
        raise ValueError("Filtered aggregate hashes are missing or inconsistent.")
    return manifest


def load_config(config_path: Path) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    config_path = config_path.resolve()
    root = find_repo_root(config_path.parent)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict):
        raise ValueError("QA config must contain a data mapping.")
    data = raw["data"]
    qa = raw.get("qa", {})
    if not isinstance(qa, dict):
        raise ValueError("QA config must contain a qa mapping.")
    source_dir = repository_path(root, data["source_dir"], "data.source_dir")
    translated_dir = repository_path(root, data["translated_dir"], "data.translated_dir")
    output_dir = repository_path(
        root,
        data.get("filtered_dir", "data/processed/ragtruth_textual_pt_nllb_filtered"),
        "data.filtered_dir",
    )
    return root, source_dir, translated_dir, output_dir, qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically filter gross NLLB translation failures")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("notebooks/configs/ragtruth_translation_qa.example.yaml"),
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing filtered output directory")
    args = parser.parse_args()
    root, source_dir, translated_dir, output_dir, qa_config = load_config(args.config)
    manifest = filter_dataset(
        root=root,
        source_dir=source_dir,
        translated_dir=translated_dir,
        output_dir=output_dir,
        qa_config=qa_config,
        force=args.force,
    )
    print(json.dumps({
        "status": manifest["status"],
        "output_dataset": manifest["output_dataset"],
        "counts": manifest["counts"],
        "run_signature": manifest["run_signature"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
