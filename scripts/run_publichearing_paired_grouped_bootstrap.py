#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.paired_grouped_bootstrap import BootstrapConfig, run_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap pareado agrupado por hearing_id no PublicHearingBR.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_bootstrap(BootstrapConfig.from_yaml(args.config), validate_only=args.validate_only, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
