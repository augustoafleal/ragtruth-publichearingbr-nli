from __future__ import annotations

import argparse
from pathlib import Path

from ragtruth_transfer.paper_reporting import generate_paper_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen PublicHearingBR paper reporting outputs.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional reporting output root; default: results/paper.")
    parser.add_argument(
        "--official-run",
        type=Path,
        default=None,
        help="Optional explicit official run path. It must resolve to fc7bbae7bd5d0c99.",
    )
    args = parser.parse_args()
    generate_paper_results(output_root=args.output_root, official_run=args.official_run)


if __name__ == "__main__":
    main()

