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
        description="Alinha o artefato EN canônico ao dataset PT e valida a integridade."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Executa a validação completa sem gravar artefatos.",
    )
    args = parser.parse_args()

    config = load_alignment_config(args.config)
    for label, path in (("en_parquet", config.en_parquet), ("pt_parquet", config.pt_parquet)):
        if not path.is_file():
            parser.error(f"{label} não encontrado: {path}")

    try:
        result = align_translation_quality(config, validate_only=args.validate_only)
    except AlignmentIntegrityError as error:
        print(json.dumps({"backend": config.backend, "signature": config.signature, "counts": error.counts.to_dict()}, ensure_ascii=False, indent=2))
        print("GATE DE INTEGRIDADE EN<->PT FALHOU.", file=sys.stderr)
        sys.exit(1)

    print(
        json.dumps(
            {
                "backend": config.backend,
                "signature": config.signature,
                "run_dir": None if args.validate_only else str(config.run_dir),
                "counts": result.counts.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
