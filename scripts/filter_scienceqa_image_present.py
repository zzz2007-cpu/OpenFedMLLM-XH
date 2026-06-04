#!/usr/bin/env python3
"""Filter ScienceQA parquet splits to samples with non-empty images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "ScienceQA" / "data"
OUTPUT_DIR = PROJECT_ROOT / "ScienceQA" / "image_present" / "data"
META_PATH = PROJECT_ROOT / "ScienceQA" / "image_present" / "summary.json"

SPLIT_FILES = {
    "train": "train-00000-of-00001-1028f23e353fbe3e.parquet",
    "validation": "validation-00000-of-00001-6c7328ff6c84284c.parquet",
    "test": "test-00000-of-00001-f0e719df791966ff.parquet",
}


def has_image(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    image_bytes = value.get("bytes")
    if image_bytes is None:
        return False
    return len(image_bytes) > 0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": "ScienceQA",
        "filter": "image is a dict and image['bytes'] is non-empty",
        "input_dir": str(INPUT_DIR.relative_to(PROJECT_ROOT)),
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "splits": {},
    }

    for split, filename in SPLIT_FILES.items():
        input_path = INPUT_DIR / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split: {input_path}")

        df = pd.read_parquet(input_path)
        mask = df["image"].map(has_image)
        filtered = df.loc[mask].reset_index(drop=True)

        output_path = OUTPUT_DIR / filename
        filtered.to_parquet(output_path, index=False)

        total = int(len(df))
        kept = int(len(filtered))
        removed = total - kept
        summary["splits"][split] = {
            "input_file": str(input_path.relative_to(PROJECT_ROOT)),
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            "total_samples": total,
            "image_present_samples": kept,
            "removed_no_image_samples": removed,
            "image_present_ratio": round(kept / total, 6) if total else 0.0,
        }
        print(f"{split}: kept {kept}/{total}, removed {removed}")

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {META_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
