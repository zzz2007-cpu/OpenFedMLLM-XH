#!/usr/bin/env python3
"""Replace local Hateful Memes with the Kaggle 10k layout and rebuild splits."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_SPLITS = {
    "train.jsonl": 8500,
    "dev.jsonl": 500,
    "test.jsonl": 1000,
}
LOG_PATH = Path("logs/hateful_memes_kaggle_replace.log")
DATA_REPORT = Path("outputs/analysis/kaggle_hateful_memes_10k_validation_report.json")
FED_REPORT = Path("outputs/analysis/hateful_memes_federated_validation_report.json")


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("downloads/kaggle_hateful_memes_raw/data"),
        help="Directory containing Kaggle train/dev/test jsonl and img/.",
    )
    parser.add_argument("--target_dir", type=Path, default=Path("hateful_memes"))
    parser.add_argument("--backup_root", type=Path, default=Path("backup"))
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_source(source_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source_dir": str(source_dir),
        "required_splits": REQUIRED_SPLITS,
        "splits": {},
        "image_file_count": 0,
        "missing_files": [],
        "missing_image_refs": [],
        "invalid_labels": [],
        "unlabeled_test_rows": 0,
        "success": False,
    }
    img_dir = source_dir / "img"
    if not img_dir.exists():
        report["missing_files"].append(str(img_dir))
        write_json(DATA_REPORT, report)
        return report

    image_files = [
        p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    report["image_file_count"] = len(image_files)

    for split, expected_count in REQUIRED_SPLITS.items():
        split_path = source_dir / split
        if not split_path.exists():
            report["missing_files"].append(str(split_path))
            continue
        rows = read_jsonl(split_path)
        label_counter: Counter[str] = Counter()
        split_report = {
            "rows": len(rows),
            "expected_rows": expected_count,
            "missing_image_refs": 0,
            "invalid_labels": 0,
            "missing_label_rows": 0,
            "label_distribution": {},
        }
        for line_no, row in enumerate(rows, 1):
            img = row.get("img")
            if not img or not (source_dir / str(img)).exists():
                split_report["missing_image_refs"] += 1
                if len(report["missing_image_refs"]) < 100:
                    report["missing_image_refs"].append(
                        {"split": split, "line": line_no, "id": row.get("id"), "img": img}
                    )
            raw_label = row.get("label")
            if raw_label is None:
                split_report["missing_label_rows"] += 1
                if split == "test.jsonl":
                    report["unlabeled_test_rows"] += 1
                    continue
            try:
                label = int(raw_label)
            except (TypeError, ValueError):
                label = -1
            if label not in {0, 1}:
                split_report["invalid_labels"] += 1
                if len(report["invalid_labels"]) < 100:
                    report["invalid_labels"].append(
                        {"split": split, "line": line_no, "id": row.get("id"), "label": row.get("label")}
                    )
            else:
                label_counter[str(label)] += 1
        split_report["label_distribution"] = dict(sorted(label_counter.items()))
        report["splits"][split] = split_report

    counts_ok = all(report["splits"].get(split, {}).get("rows") == count for split, count in REQUIRED_SPLITS.items())
    refs_ok = not report["missing_image_refs"]
    labels_ok = not report["invalid_labels"]
    report["success"] = (
        not report["missing_files"]
        and counts_ok
        and refs_ok
        and labels_ok
        and report["image_file_count"] >= 10000
    )
    write_json(DATA_REPORT, report)
    return report


def replace_dataset(source_dir: Path, target_dir: Path, backup_root: Path) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"hateful_memes_before_kaggle_10k_{timestamp}"
    old_backup = backup_dir / "hateful_memes"
    backup_root.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        log(f"Moving old dataset {target_dir} -> {old_backup}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_dir), str(old_backup))
    else:
        log(f"No existing {target_dir}; creating new dataset without backup")

    log(f"Copying Kaggle dataset {source_dir} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_dir / "img", target_dir / "img")
    for split in REQUIRED_SPLITS:
        shutil.copy2(source_dir / split, target_dir / split)

    # Compatibility aliases for existing configs that default to dev_seen/test_seen.
    shutil.copy2(source_dir / "dev.jsonl", target_dir / "dev_seen.jsonl")
    shutil.copy2(source_dir / "test.jsonl", target_dir / "test_seen.jsonl")

    manifest = {
        "source": "KaggleHub parthplc/facebook-hateful-meme-dataset",
        "source_dir": str(source_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Kaggle 10k layout. dev_seen.jsonl and test_seen.jsonl are compatibility aliases for dev.jsonl/test.jsonl.",
        "splits": {
            "train.jsonl": 8500,
            "dev.jsonl": 500,
            "test.jsonl": 1000,
            "dev_seen.jsonl": "alias of dev.jsonl",
            "test_seen.jsonl": "alias of test.jsonl",
        },
    }
    write_json(target_dir / "KAGGLE_SOURCE_MANIFEST.json", manifest)
    return backup_dir if old_backup.exists() else None


def run_split_generator(target_dir: Path, num_clients: int, seed: int) -> None:
    cmd = [
        sys.executable,
        "scripts/build_hateful_memes_federated.py",
        "--data_dir",
        str(target_dir),
        "--out_dir",
        str(target_dir / "federated"),
        "--num_clients",
        str(num_clients),
        "--seed",
        str(seed),
        "--alphas",
        "1.0",
        "0.5",
        "0.1",
        "--missing_rates",
        "0.3",
        "0.4",
        "0.5",
        "--cross_ratios",
        "3:7",
        "5:5",
        "7:3",
        "--hybrid_keep_probs",
        "0.8",
        "0.7",
        "0.6",
    ]
    log("Generating federated splits: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_federated(target_dir: Path) -> dict[str, Any]:
    fed_root = target_dir / "federated"
    report: dict[str, Any] = {
        "federated_root": str(fed_root),
        "settings": {},
        "errors": [],
        "success": False,
    }
    for setting_dir in sorted(p for p in fed_root.glob("*/*") if p.is_dir()):
        key = setting_dir.relative_to(fed_root).as_posix()
        setting = {
            "total_rows": 0,
            "client_files": 0,
            "empty_clients": [],
            "missing_images": 0,
            "both_modalities_missing": 0,
            "invalid_labels": 0,
            "aligned_bad_modalities": 0,
        }
        for client_file in sorted(setting_dir.glob("client_*.jsonl")):
            setting["client_files"] += 1
            rows = read_jsonl(client_file)
            if not rows:
                setting["empty_clients"].append(client_file.name)
            setting["total_rows"] += len(rows)
            for line_no, row in enumerate(rows, 1):
                img = row.get("img")
                text = row.get("text")
                has_img = img is not None and str(img) != ""
                has_text = text is not None and str(text) != ""
                if not has_img and not has_text:
                    setting["both_modalities_missing"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: both img and text missing")
                if has_img and not (target_dir / str(img)).exists():
                    setting["missing_images"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: missing image {img}")
                try:
                    label = int(row.get("label"))
                except (TypeError, ValueError):
                    label = -1
                if label not in {0, 1}:
                    setting["invalid_labels"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: invalid label {row.get('label')!r}")
                if setting_dir.name == "aligned" and (not has_img or not has_text):
                    setting["aligned_bad_modalities"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: aligned sample missing modality")

        if setting["total_rows"] != 8500:
            report["errors"].append(f"{key}: total_rows={setting['total_rows']} != 8500")
        if setting["client_files"] != 10:
            report["errors"].append(f"{key}: client_files={setting['client_files']} != 10")
        if setting["empty_clients"]:
            report["errors"].append(f"{key}: empty_clients={setting['empty_clients']}")

        meta_path = setting_dir / "meta.json"
        summary_path = setting_dir / "summary.json"
        if meta_path.exists() and not summary_path.exists():
            shutil.copy2(meta_path, summary_path)
        report["settings"][key] = setting

    report["success"] = not report["errors"]
    write_json(FED_REPORT, report)
    return report


def main() -> None:
    args = parse_args()
    log("Starting Kaggle 10k Hateful Memes replacement")
    log(f"Source dir: {args.source_dir}")
    report = validate_source(args.source_dir)
    log(f"Validation success: {report['success']}")
    if not report["success"]:
        log("Validation failed. Existing dataset was not touched.")
        raise SystemExit(2)

    backup_dir = replace_dataset(args.source_dir, args.target_dir, args.backup_root)
    log(f"Backup dir: {backup_dir}")
    run_split_generator(args.target_dir, args.num_clients, args.seed)
    fed_report = validate_federated(args.target_dir)
    log(f"Federated validation success: {fed_report['success']}")
    if not fed_report["success"]:
        raise SystemExit(3)
    log("Completed Kaggle 10k Hateful Memes replacement")


if __name__ == "__main__":
    main()
