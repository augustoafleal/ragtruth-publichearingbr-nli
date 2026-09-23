#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ragtruth_transfer.translation_quality.config import load_scoring_config
from ragtruth_transfer.translation_quality.scoring import score_translation_quality


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pontua a qualidade da tradução (heurísticas, COMETKiwi, NLI-consistency)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Sobrescreve scoring.sample_limit.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valida config e insumos sem carregar modelos.",
    )
    args = parser.parse_args()

    config = load_scoring_config(args.config)
    if args.limit is not None:
        config = replace(config, sample_limit=args.limit)

    if args.validate_only:
        if not config.aligned_parquet.is_file():
            parser.error(
                f"aligned_parquet não encontrado: {config.aligned_parquet}. "
                "Execute o Stage 0 (align_translation_quality.py) primeiro."
            )
        rows = len(pd.read_parquet(config.aligned_parquet, columns=["example_id"]))
        print(
            json.dumps(
                {
                    "backend": config.backend,
                    "signature": config.signature,
                    "metrics": list(config.metrics),
                    "aligned_parquet": str(config.aligned_parquet),
                    "aligned_rows": rows,
                    "sample_split": config.sample_split,
                    "sample_limit": config.sample_limit,
                    "device": config.device,
                    "batch_size": config.batch_size,
                    "validate_only": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result = score_translation_quality(config)
    print(
        json.dumps(
            {
                "backend": config.backend,
                "signature": config.signature,
                "run_dir": str(config.run_dir),
                "summary": result.summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
