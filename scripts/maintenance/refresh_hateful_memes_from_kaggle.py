#!/usr/bin/env python3
"""Validate Kaggle Hateful Memes data, replace local data safely, and rebuild splits."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "train.jsonl": 8500,
    "dev_seen.jsonl": 500,
    "test_seen.jsonl": 1000,
    "dev_unseen.jsonl": 540,
    "test_unseen.jsonl": 2000,
}
TOTAL_EXPECTED = sum(EXPECTED_COUNTS.values())
LOG_PATH = Path("logs/hateful_memes_kaggle_refresh.log")
VALIDATION_REPORT = Path("outputs/analysis/kaggle_hateful_memes_validation_report.json")
FEDERATED_REPORT = Path("outputs/analysis/hateful_memes_federated_validation_report.json")


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle_path", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, default=Path("tmp/hateful_memes_kaggle_extract"))
    parser.add_argument("--target_dir", type=Path, default=Path("hateful_memes"))
    parser.add_argument("--backup_root", type=Path, default=Path("backup"))
    parser.add_argument("--skip_replace", action="store_true")
    parser.add_argument("--skip_generate", action="store_true")
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


def safe_extract_zip(zip_path: Path, dest: Path) -> None:
    log(f"Extracting zip safely: {zip_path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(dest)


def prepare_search_root(kaggle_path: Path, work_dir: Path) -> Path:
    if not kaggle_path.exists():
        raise FileNotFoundError(f"Kaggle path does not exist: {kaggle_path}")

    archives = sorted([*kaggle_path.rglob("*.zip")])
    if archives:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            safe_extract_zip(archive, work_dir / archive.stem)
        return work_dir
    return kaggle_path


def find_dataset_root(search_root: Path) -> tuple[Path, Path]:
    candidates: list[tuple[int, Path, Path]] = []
    for train_path in search_root.rglob("train.jsonl"):
        root = train_path.parent
        present = sum(1 for name in EXPECTED_COUNTS if (root / name).exists())
        img_dirs = [p for p in (root / "img", root / "images") if p.exists() and p.is_dir()]
        if img_dirs:
            candidates.append((present, root, img_dirs[0]))

    if not candidates:
        raise FileNotFoundError(f"Could not find train.jsonl with img/ or images/ under {search_root}")

    candidates.sort(key=lambda item: (-item[0], len(str(item[1]))))
    _, root, img_dir = candidates[0]
    return root, img_dir


def validate_dataset(dataset_root: Path, img_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "image_dir": str(img_dir),
        "expected_counts": EXPECTED_COUNTS,
        "splits": {},
        "total_rows": 0,
        "image_file_count": 0,
        "missing_jsonl_files": [],
        "missing_image_references": [],
        "invalid_labels": [],
        "success": False,
    }

    image_files = [p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    report["image_file_count"] = len(image_files)

    for split_name, expected_count in EXPECTED_COUNTS.items():
        split_path = dataset_root / split_name
        if not split_path.exists():
            report["missing_jsonl_files"].append(split_name)
            continue

        rows = read_jsonl(split_path)
        split_report = {
            "rows": len(rows),
            "expected_rows": expected_count,
            "label_distribution": {},
            "missing_images": 0,
            "invalid_labels": 0,
            "has_label_rows": 0,
            "missing_label_rows": 0,
        }
        labels = Counter()
        for idx, row in enumerate(rows, 1):
            img = row.get("img")
            if img:
                image_path = dataset_root / str(img)
                if not image_path.exists() and img_dir.name != "img":
                    image_path = img_dir / Path(str(img)).name
                if not image_path.exists():
                    split_report["missing_images"] += 1
                    if len(report["missing_image_references"]) < 100:
                        report["missing_image_references"].append(
                            {"split": split_name, "line": idx, "id": row.get("id"), "img": img}
                        )
            label = row.get("label")
            if label is None:
                split_report["missing_label_rows"] += 1
            else:
                split_report["has_label_rows"] += 1
                try:
                    label_int = int(label)
                except (TypeError, ValueError):
                    label_int = -1
                if label_int not in {0, 1}:
                    split_report["invalid_labels"] += 1
                    if len(report["invalid_labels"]) < 100:
                        report["invalid_labels"].append(
                            {"split": split_name, "line": idx, "id": row.get("id"), "label": label}
                        )
                else:
                    labels[str(label_int)] += 1
        split_report["label_distribution"] = dict(sorted(labels.items()))
        report["splits"][split_name] = split_report
        report["total_rows"] += len(rows)

    all_counts_match = all(
        report["splits"].get(name, {}).get("rows") == expected
        for name, expected in EXPECTED_COUNTS.items()
    )
    no_missing_images = not report["missing_image_references"] and all(
        split.get("missing_images", 0) == 0 for split in report["splits"].values()
    )
    no_invalid_labels = not report["invalid_labels"] and all(
        split.get("invalid_labels", 0) == 0 for split in report["splits"].values()
    )
    report["success"] = (
        not report["missing_jsonl_files"]
        and all_counts_match
        and no_missing_images
        and no_invalid_labels
        and report["total_rows"] == TOTAL_EXPECTED
        and report["image_file_count"] >= TOTAL_EXPECTED
    )
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    log(f"Wrote report: {path}")


def copy_clean_dataset(dataset_root: Path, img_dir: Path, target_dir: Path, backup_root: Path) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"hateful_memes_before_kaggle_{timestamp}"
    backup_data_dir = backup_dir / "hateful_memes"
    backup_root.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        log(f"Moving current {target_dir} -> {backup_data_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_dir), str(backup_data_dir))
    else:
        log(f"No existing {target_dir} found; no backup move needed")

    log(f"Creating clean target dir: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(img_dir, target_dir / "img")
    for name in EXPECTED_COUNTS:
        shutil.copy2(dataset_root / name, target_dir / name)
    for extra_name in ("LICENSE", "LICENSE.txt", "README", "README.md"):
        extra = dataset_root / extra_name
        if extra.exists() and extra.is_file():
            shutil.copy2(extra, target_dir / extra.name)
    return backup_dir if backup_data_dir.exists() else None


def run_split_generator() -> None:
    cmd = [
        sys.executable,
        "scripts/build_hateful_memes_federated.py",
        "--data_dir",
        "hateful_memes",
        "--out_dir",
        "hateful_memes/federated",
        "--num_clients",
        "10",
        "--seed",
        "42",
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
    log("Running federated split generator: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_federated(root: Path = Path("hateful_memes")) -> dict[str, Any]:
    fed_root = root / "federated"
    report: dict[str, Any] = {
        "federated_root": str(fed_root),
        "settings": {},
        "errors": [],
        "success": False,
    }
    for setting_dir in sorted(p for p in fed_root.glob("*/*") if p.is_dir()):
        client_files = sorted(setting_dir.glob("client_*.jsonl"))
        setting_key = str(setting_dir.relative_to(fed_root)).replace("\\", "/")
        setting_report = {
            "client_files": len(client_files),
            "total_rows": 0,
            "empty_clients": [],
            "missing_images": 0,
            "invalid_labels": 0,
            "both_modalities_missing": 0,
            "aligned_bad_modalities": 0,
        }
        for client_file in client_files:
            rows = read_jsonl(client_file)
            if not rows:
                setting_report["empty_clients"].append(client_file.name)
            setting_report["total_rows"] += len(rows)
            for line_no, row in enumerate(rows, 1):
                img = row.get("img")
                text = row.get("text")
                label = row.get("label")
                has_img = img is not None and str(img) != ""
                has_text = text is not None and str(text) != ""
                if not has_img and not has_text:
                    setting_report["both_modalities_missing"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: img and text are both missing")
                if label not in {0, 1}:
                    try:
                        ok_label = int(label) in {0, 1}
                    except (TypeError, ValueError):
                        ok_label = False
                    if not ok_label:
                        setting_report["invalid_labels"] += 1
                        report["errors"].append(f"{client_file}:{line_no}: invalid label {label!r}")
                if has_img and not (root / str(img)).exists():
                    setting_report["missing_images"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: missing image {img}")
                if setting_dir.name == "aligned" and (not has_img or not has_text):
                    setting_report["aligned_bad_modalities"] += 1
                    report["errors"].append(f"{client_file}:{line_no}: aligned sample missing modality")
        if setting_report["total_rows"] != 8500:
            report["errors"].append(f"{setting_key}: total rows {setting_report['total_rows']} != 8500")
        if setting_report["empty_clients"]:
            report["errors"].append(f"{setting_key}: empty clients {setting_report['empty_clients']}")
        report["settings"][setting_key] = setting_report

        meta_path = setting_dir / "meta.json"
        summary_path = setting_dir / "summary.json"
        if meta_path.exists() and not summary_path.exists():
            shutil.copy2(meta_path, summary_path)
            log(f"Copied {meta_path} -> {summary_path}")

    report["success"] = not report["errors"]
    return report


def main() -> None:
    args = parse_args()
    log("Starting Kaggle Hateful Memes refresh")
    log(f"Kaggle path: {args.kaggle_path}")

    search_root = prepare_search_root(args.kaggle_path, args.work_dir)
    dataset_root, img_dir = find_dataset_root(search_root)
    log(f"Detected dataset root: {dataset_root}")
    log(f"Detected image dir: {img_dir}")

    validation_report = validate_dataset(dataset_root, img_dir)
    write_report(VALIDATION_REPORT, validation_report)
    if not validation_report["success"]:
        log("Validation failed. Leaving existing hateful_memes untouched.")
        raise SystemExit(2)

    backup_dir: Path | None = None
    if args.skip_replace:
        log("--skip_replace set; validation succeeded but no data replacement performed.")
    else:
        backup_dir = copy_clean_dataset(dataset_root, img_dir, args.target_dir, args.backup_root)
        log(f"Backup path: {backup_dir}")

    if args.skip_generate:
        log("--skip_generate set; federated split generation skipped.")
    else:
        run_split_generator()
        fed_report = validate_federated(args.target_dir)
        write_report(FEDERATED_REPORT, fed_report)
        if not fed_report["success"]:
            log("Federated validation failed after generation.")
            raise SystemExit(3)

    log("Completed Kaggle Hateful Memes refresh successfully")


if __name__ == "__main__":
    main()
