#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragtruth_transfer.publichearing.campaign import aggregate_experiment, prepare_experiment, train_experiment
from ragtruth_transfer.publichearing.config import load_publichearing_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa uma campanha PublicHearingBR-NLI completa.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force-tokenization", action="store_true")
    args = parser.parse_args()
    prepared = prepare_experiment(load_publichearing_config(args.config, args.overrides), args.force_tokenization)
    train_experiment(prepared)
    print(json.dumps(aggregate_experiment(prepared), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
