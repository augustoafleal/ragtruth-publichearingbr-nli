#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.translation import TranslationConfig, translate_ragtruth


def main() -> None:
    parser = argparse.ArgumentParser(description="Traduz o RAGTruth processado preservando seu schema.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma uma execução compatível usando o cache e outputs já concluídos.",
    )
    args = parser.parse_args()
    manifest = translate_ragtruth(TranslationConfig.from_yaml(args.config), resume=args.resume)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
