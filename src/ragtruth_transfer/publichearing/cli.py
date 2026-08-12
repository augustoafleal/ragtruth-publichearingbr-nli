from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import aggregate_experiment, prepare_experiment, train_experiment, validate_run
from .config import load_publichearing_config


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Sobrescreve uma chave YAML (repetível).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Campanha in-domain PublicHearingBR-NLI com CV agrupada.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Baixa, normaliza, divide e pré-tokeniza o dataset."); _config_args(prepare); prepare.add_argument("--force-tokenization", action="store_true")
    train = commands.add_parser("train", help="Treina folds/seeds; resultados completos são reutilizados."); _config_args(train); train.add_argument("--fold", type=int); train.add_argument("--seed", type=int)
    aggregate = commands.add_parser("aggregate", help="Exige todos os folds/seeds e gera OOF/bootstrap/artefatos."); _config_args(aggregate)
    validate = commands.add_parser("validate-run", help="Valida cobertura OOF e isolamento por audiência."); validate.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate-run":
        result = validate_run(args.run_dir)
    else:
        config = load_publichearing_config(args.config, args.overrides)
        prepared = prepare_experiment(config, force_tokenization=getattr(args, "force_tokenization", False))
        if args.command == "prepare":
            result = {"run_dir": str(prepared.run_dir), "signature": prepared.signature, "cache": str(prepared.cache_path), "examples": len(prepared.frame)}
        elif args.command == "train":
            rows = train_experiment(prepared, args.fold, args.seed)
            result = {"run_dir": str(prepared.run_dir), "completed_seed_runs": len(rows), "reused": sum(bool(row["reused"]) for row in rows)}
        else:
            result = aggregate_experiment(prepared)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
