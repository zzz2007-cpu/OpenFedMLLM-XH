#!/usr/bin/env python3
"""Validate Hateful Memes train/dev protocol for MiniCPM-V experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def validate_split(root: Path, split_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    path = root / split_name
    errors = []
    if not path.exists():
        return [], [f"missing split file: {path}"]
    rows = read_jsonl(path)
    required = {"id", "img", "text", "label"}
    for row in rows:
        line_no = row["_line_no"]
        sample_id = row.get("id")
        missing_fields = sorted(k for k in required if k not in row or row.get(k) is None)
        if missing_fields:
            errors.append(f"{path}:{line_no}: id={sample_id!r}: missing fields {missing_fields}")
            continue
        try:
            label = int(row.get("label"))
        except (TypeError, ValueError):
            label = -1
        if label not in {0, 1}:
            errors.append(f"{path}:{line_no}: id={sample_id!r}: invalid label {row.get('label')!r}")
        img_path = root / str(row.get("img"))
        if not img_path.exists():
            errors.append(f"{path}:{line_no}: id={sample_id!r}: missing image {img_path}")
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("hateful_memes"))
    parser.add_argument("--report", type=Path, default=Path("outputs/analysis/hateful_memes_dataset_check.json"))
    args = parser.parse_args()

    root = args.data_root
    all_errors = []
    train_rows, train_errors = validate_split(root, "train.jsonl")
    dev_rows, dev_errors = validate_split(root, "dev.jsonl")
    all_errors.extend(train_errors)
    all_errors.extend(dev_errors)

    train_ids = {str(row.get("id")) for row in train_rows}
    dev_ids = {str(row.get("id")) for row in dev_rows}
    train_imgs = {str(row.get("img")) for row in train_rows}
    dev_imgs = {str(row.get("img")) for row in dev_rows}
    id_overlap = sorted(train_ids & dev_ids)
    img_overlap = sorted(train_imgs & dev_imgs)
    for sample_id in id_overlap[:50]:
        all_errors.append(f"train/dev id overlap: {sample_id}")
    for img in img_overlap[:50]:
        all_errors.append(f"train/dev image overlap: {img}")

    report = {
        "data_root": str(root),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "missing_images": sum(1 for err in all_errors if "missing image" in err),
        "id_overlap_count": len(id_overlap),
        "image_overlap_count": len(img_overlap),
        "success": not all_errors,
        "errors": all_errors[:200],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
