#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.ragtruth_training_view import (
    SUPPORTED_POLICY,
    build_training_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and build a deduplicated RAGTruth QA training view")
    parser.add_argument("--input-run-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/ragtruth_qa_training_view"))
    parser.add_argument("--policy", default=SUPPORTED_POLICY)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, help="Reservado para lineage; o input-run-dir continua sendo obrigatório quando há múltiplos runs.")
    parser.add_argument("--raw-response-path", type=Path)
    parser.add_argument("--raw-source-path", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.config and not args.config.is_file():
        parser.error(f"Configuração não encontrada: {args.config}")
    manifest = build_training_view(
        args.input_run_dir,
        output_root=args.output_root,
        policy=args.policy,
        audit_only=args.audit_only,
        resume=args.resume,
        force=args.force,
        raw_response_path=args.raw_response_path,
        raw_source_path=args.raw_source_path,
        results_root=args.results_root,
        seed=args.seed,
    )
    print(json.dumps({"status": manifest.get("status"), "signature": manifest["signature"], "output_dir": str((args.output_root / args.policy / manifest["signature"]).resolve()), "parent": manifest["parent"], "counts": manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
