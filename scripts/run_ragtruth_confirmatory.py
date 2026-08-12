#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.ragtruth_confirmatory import ConfirmatoryConfig, run_confirmatory


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestra a campanha confirmatória RAGTruth de três seeds.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["train", "evaluate", "aggregate"], default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_confirmatory(ConfirmatoryConfig.from_yaml(args.config), args.phase, resume=args.resume, validate=args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
