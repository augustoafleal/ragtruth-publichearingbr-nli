#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragtruth_transfer.translation_quality.alignment import (
    AlignmentIntegrityError,
    align_translation_quality,
)
from ragtruth_transfer.translation_quality.config import load_alignment_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alinha o artefato source canônico ao dataset translated e valida a integridade."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Executa a validação completa sem gravar artefatos; "
            "omita esta opção para gerar aligned.parquet."
        ),
    )
    args = parser.parse_args()

    config = load_alignment_config(args.config)
    for label, path in (
        ("source_artifact", config.source_artifact),
        ("translated_artifact", config.translated_artifact),
    ):
        if not path.is_file():
            parser.error(f"{label} não encontrado: {path}")

    try:
        result = align_translation_quality(config, validate_only=args.validate_only)
    except AlignmentIntegrityError as error:
        print(json.dumps({"backend": config.backend, "signature": config.signature, "counts": error.counts.to_dict()}, ensure_ascii=False, indent=2))
        print("GATE DE INTEGRIDADE source→target FALHOU.", file=sys.stderr)
        sys.exit(1)

    print(
        json.dumps(
            {
                "backend": config.backend,
                "signature": config.signature,
                "run_dir": str(config.run_dir),
                "aligned_parquet": str(config.run_dir / "aligned.parquet"),
                "validate_only": args.validate_only,
                "artifacts_written": not args.validate_only,
                "counts": result.counts.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
