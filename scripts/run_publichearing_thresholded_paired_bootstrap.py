#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.thresholded_paired_grouped_bootstrap import (
    GenericThresholdedBootstrapConfig,
    ThresholdedBootstrapConfig,
    load_thresholded_config,
    run_generic_thresholded_bootstrap,
    run_thresholded_bootstrap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap pareado e agrupado, com thresholds congelados.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_thresholded_config(args.config)
    if isinstance(config, GenericThresholdedBootstrapConfig):
        if args.resume:
            raise SystemExit("--resume ainda não é suportado no modo genérico; a saída nunca é sobrescrita.")
        result = run_generic_thresholded_bootstrap(config, validate_only=args.validate_only)
    else:
        result = run_thresholded_bootstrap(config, validate_only=args.validate_only, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
