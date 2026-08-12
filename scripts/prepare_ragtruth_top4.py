#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ragtruth_transfer.ragtruth_top4_config import load_top4_config
from ragtruth_transfer.ragtruth_top4_pipeline import estimate_top4, prepare_top4


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGTruth QA -> claim + exactly four semantic evidence slots")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=["train", "test"], help="Pode ser repetido; por padrão usa os splits da configuração.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Dispositivo do encoder; não usado em --estimate.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0, help="Reservado para compatibilidade; a implementação atual usa inferência determinística em um processo.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--estimate", action="store_true", help="Estima claims/chunks/cache sem carregar encoder ou calcular embeddings.")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers deve ser >= 0")
    config = load_top4_config(args.config)
    if args.cache_dir:
        config = replace(config, cache_root=args.cache_dir.resolve())
    split_filter = set(args.split) if args.split else None
    if args.estimate:
        print(json.dumps(estimate_top4(config, split_filter=split_filter, limit=args.limit), ensure_ascii=False, indent=2))
        return
    manifest = prepare_top4(
        config,
        output_dir=args.output_dir,
        split_filter=split_filter,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        resume=args.resume,
        force=args.force,
    )
    resolved_output = args.output_dir.resolve() if args.output_dir else (config.output_root / config.run_name / manifest["signature"]).resolve()
    print(json.dumps({"status": "completed", "signature": manifest["signature"], "output_dir": str(resolved_output), "counts": manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
