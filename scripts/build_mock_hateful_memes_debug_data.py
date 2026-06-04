#!/usr/bin/env python3
"""Build a tiny Hateful Memes-style dataset for CPU FedAvg/FedCHI smoke tests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tiny mock Hateful Memes data and federated splits.")
    parser.add_argument("--out_dir", type=Path, default=Path("tmp/mock_hateful_memes"))
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_records(split: str, count: int) -> list[dict]:
    records = []
    for idx in range(count):
        label = idx % 2
        cue = "hateful attack" if label == 1 else "friendly comment"
        records.append({
            "id": f"{split}_{idx:03d}",
            "img": f"img/{split}_{idx:03d}.jpg",
            "text": f"{cue} sample {idx}",
            "label": label,
        })
    return records


def write_placeholder_images(root: Path, records: Iterable[dict]) -> None:
    img_dir = root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = root / record["img"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"mock-image-placeholder\n")


def run_partition_builder(root: Path, num_clients: int, seed: int) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "build_hateful_memes_federated.py"
    cmd = [
        sys.executable,
        str(script),
        "--data_dir",
        str(root),
        "--out_dir",
        str(root / "federated"),
        "--num_clients",
        str(num_clients),
        "--seed",
        str(seed),
        "--alphas",
        "0.5",
        "--missing_rates",
        "0.3",
        "--cross_ratios",
        f"{num_clients // 2}:{num_clients - (num_clients // 2)}",
        "--hybrid_keep_probs",
        "0.8",
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main() -> None:
    args = parse_args()
    if args.num_clients != 2:
        raise ValueError("This mock debug dataset is intentionally fixed to --num_clients 2.")

    root = args.out_dir
    train_records = build_records("train", 12)
    dev_records = build_records("dev", 6)
    write_placeholder_images(root, [*train_records, *dev_records])
    write_jsonl(root / "train.jsonl", train_records)
    write_jsonl(root / "dev.jsonl", dev_records)
    run_partition_builder(root=root, num_clients=args.num_clients, seed=args.seed)

    summary_path = root / "mock_debug_summary.json"
    summary = {
        "root": str(root),
        "train_jsonl": str(root / "train.jsonl"),
        "dev_jsonl": str(root / "dev.jsonl"),
        "federated_summary": str(root / "federated" / "summary.json"),
        "expected_smoke_settings": [
            "federated/iid/aligned",
            "federated/dir_0.5/cross_1_1",
        ],
        "num_train": len(train_records),
        "num_dev": len(dev_records),
        "num_clients": int(args.num_clients),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
