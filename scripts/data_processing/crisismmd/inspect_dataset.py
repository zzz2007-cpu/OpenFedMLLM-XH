#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from crisismmd.constants import STATS_FILENAME, VALIDATION_FILENAME
    from crisismmd.io_utils import write_json
    from crisismmd.stats import summarize_dataset
else:
    from .constants import STATS_FILENAME, VALIDATION_FILENAME
    from .io_utils import write_json
    from .stats import summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a standardized or federated CrisisMMD dataset directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory to inspect.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    summary = summarize_dataset(dataset_dir, path_checks=True)
    write_json(dataset_dir / STATS_FILENAME, summary["stats"])
    write_json(dataset_dir / VALIDATION_FILENAME, summary["validation"])
    print(f"Stats written to {dataset_dir / STATS_FILENAME}")
    print(f"Validation written to {dataset_dir / VALIDATION_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
