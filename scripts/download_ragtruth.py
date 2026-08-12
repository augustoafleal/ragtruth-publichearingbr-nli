#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FILES = {
    "response.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
    "source_info.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa os arquivos oficiais do RAGTruth.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        destination = args.output_dir / filename
        if destination.exists() and not args.force:
            print(f"Mantido: {destination}")
            continue
        print(f"Baixando {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"Salvo: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
