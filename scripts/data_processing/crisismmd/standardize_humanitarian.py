#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from crisismmd.constants import (
        DEFAULT_IMAGE_PREFIX,
        DEFAULT_INPUT_ROOT,
        DEFAULT_LABEL_SOURCE,
        MANIFEST_FILENAME,
        STANDARD_SPLIT_FILES,
    )
    from crisismmd.dataset_utils import read_standardized_tsv
    from crisismmd.io_utils import ensure_dir, write_json
    from crisismmd.stats import write_dataset_reports
else:
    from .constants import (
        DEFAULT_IMAGE_PREFIX,
        DEFAULT_INPUT_ROOT,
        DEFAULT_LABEL_SOURCE,
        MANIFEST_FILENAME,
        STANDARD_SPLIT_FILES,
    )
    from .dataset_utils import read_standardized_tsv
    from .io_utils import ensure_dir, write_json
    from .stats import write_dataset_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize CrisisMMD humanitarian TSV splits into a reusable JSON format."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing task_humanitarian_text_img_{train,dev,test}.tsv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Standardized dataset output directory.",
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        default=DEFAULT_IMAGE_PREFIX,
        help="Relative image prefix used to build training-compatible image_path fields.",
    )
    parser.add_argument(
        "--label_source",
        type=str,
        default=DEFAULT_LABEL_SOURCE,
        choices=("label", "label_text", "label_image"),
        help="Which TSV label column becomes the benchmark supervision label. Default uses label_image to match existing client shards.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    split_to_records: Dict[str, List[Dict]] = {}
    split_files = {}
    source_files = {}
    for split in ("train", "dev", "test"):
        tsv_path = input_root / f"task_humanitarian_text_img_{split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Missing input split file: {tsv_path}")
        records = read_standardized_tsv(
            tsv_path=tsv_path,
            split=split,
            image_prefix=args.image_prefix,
            label_source=args.label_source,
        )
        split_to_records[split] = records
        out_name = STANDARD_SPLIT_FILES[split]
        split_files[split] = out_name
        source_files[split] = str(tsv_path).replace("\\", "/")
        write_json(output_dir / out_name, records)

    manifest = {
        "dataset_type": "standardized_humanitarian",
        "task": "humanitarian",
        "label_source": args.label_source,
        "image_prefix": args.image_prefix,
        "input_root": str(input_root).replace("\\", "/"),
        "source_files": source_files,
        "split_files": split_files,
        "notes": [
            "Humanitarian supervision defaults to label_image for compatibility with the repository's existing client_*.json shards.",
            "The standardized format preserves cleaned text, original labels, split, and training-compatible relative image_path fields.",
        ],
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    write_dataset_reports(output_dir)
    print(f"Standardized humanitarian dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

