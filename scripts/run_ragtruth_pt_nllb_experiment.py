#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig, run_confirmatory
try:
    from scripts.filter_ragtruth_translation_failures import (
        filter_dataset,
        load_config as load_filter_config,
        validate_filtered_dataset,
    )
except ModuleNotFoundError:
    from filter_ragtruth_translation_failures import (
        filter_dataset,
        load_config as load_filter_config,
        validate_filtered_dataset,
    )


def _load_raw_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid experiment config: {path}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PT NLLB filtered RAGTruth experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml"),
    )
    parser.add_argument("--filter-config", type=Path, default=None)
    parser.add_argument("--force-filter", action="store_true", help="Rebuild the filtered output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never load model weights or train")
    args = parser.parse_args()

    experiment_path = args.config.resolve()
    raw = _load_raw_config(experiment_path)
    filter_value = args.filter_config or raw.get("filter_config", "../notebooks/configs/ragtruth_translation_qa.example.yaml")
    filter_path = Path(str(filter_value)).expanduser()
    if not filter_path.is_absolute():
        filter_path = (experiment_path.parent / filter_path).resolve()
    root, source_dir, translated_dir, output_dir, qa_config = load_filter_config(filter_path)

    if output_dir.exists() and not args.force_filter:
        filter_manifest = validate_filtered_dataset(
            root=root,
            source_dir=source_dir,
            translated_dir=translated_dir,
            output_dir=output_dir,
            qa_config=qa_config,
        )
        filter_action = "reused"
    else:
        filter_manifest = filter_dataset(
            root=root,
            source_dir=source_dir,
            translated_dir=translated_dir,
            output_dir=output_dir,
            qa_config=qa_config,
            force=args.force_filter,
        )
        filter_action = "rebuilt"

    config = ConfirmatoryConfig.from_yaml(experiment_path)
    validation = run_confirmatory(config, "train", validate=True)
    result: dict[str, Any] = {
        "status": "dry_run_valid" if args.dry_run else "validated",
        "filter_action": filter_action,
        "filter_manifest": {
            "schema_version": filter_manifest["schema_version"],
            "run_signature": filter_manifest["run_signature"],
            "dataset_sha256": filter_manifest["dataset_sha256"],
            "split_signature": filter_manifest["split_signature"],
            "counts": filter_manifest["counts"],
        },
        "training_validation": validation,
    }
    if not args.dry_run:
        result["training"] = run_confirmatory(config, "train", resume=args.resume)
        result["status"] = "trained"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
