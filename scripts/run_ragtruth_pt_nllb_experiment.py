#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig, run_confirmatory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical PT NLLB RAGTruth confirmatory campaign")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ragtruth_pt_nllb_filtered_lora_attention_mil_confirmatory.yaml"),
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "aggregate"), default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never load model weights or train")
    args = parser.parse_args()

    config = ConfirmatoryConfig.from_yaml(args.config.resolve())
    result = run_confirmatory(config, args.phase, resume=args.resume, validate=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
