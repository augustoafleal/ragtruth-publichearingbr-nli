#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.translation_quality.config import load_link_config
from ragtruth_transfer.translation_quality.linkage import (
    LinkValidationError,
    link_quality_to_detection,
    load_protocol_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relaciona a qualidade da tradução com o desempenho do detector (descritivo)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valida config e insumos sem executar a análise.",
    )
    args = parser.parse_args()

    config = load_link_config(args.config)
    for label, path in (
        ("example_scores_parquet", config.example_scores_parquet),
        ("predictions_parquet", config.predictions_parquet),
        ("protocol_thresholds_json", config.thresholds_json),
    ):
        if not path.is_file():
            parser.error(f"{label} não encontrado: {path}")

    threshold, record = load_protocol_threshold(config)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "backend": config.backend,
                    "signature": config.signature,
                    "example_scores_parquet": str(config.example_scores_parquet),
                    "predictions_parquet": str(config.predictions_parquet),
                    "protocol_thresholds_json": str(config.thresholds_json),
                    "threshold_criterion": config.threshold_criterion,
                    "threshold": threshold,
                    "quality_signal": config.quality_signal,
                    "quality_direction": config.quality_direction,
                    "num_quality_buckets": config.num_quality_buckets,
                    "run_dir": str(config.run_dir),
                    "validate_only": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    try:
        result = link_quality_to_detection(config)
    except LinkValidationError as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "backend": config.backend,
                "signature": config.signature,
                "run_dir": str(config.run_dir),
                "threshold": result.threshold,
                "summary": result.summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
