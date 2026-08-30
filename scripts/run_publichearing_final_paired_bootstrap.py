from __future__ import annotations

import argparse
from pathlib import Path

from ragtruth_transfer.final_paired_bootstrap import FinalBootstrapConfig, run_final_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final read-only paired grouped bootstrap")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_final_bootstrap(FinalBootstrapConfig.from_yaml(args.config), validate_only=args.validate_only)
    print(result)


if __name__ == "__main__":
    main()
