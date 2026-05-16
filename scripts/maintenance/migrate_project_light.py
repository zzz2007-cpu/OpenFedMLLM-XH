#!/usr/bin/env python3
"""
Low-risk project migration helper for OpenFedMLLM.

Design goals:
1) Create target folders.
2) Copy (never delete) selected logs/analysis/partitions.
3) Keep compatibility by preserving all original source paths.
4) Skip heavy or uncertain items.
5) Emit a migration report for audit and rollback.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CopyItem:
    src: str
    dst: str
    reason: str
    mode: str = "all"  # all | lightweight_federated
    optional: bool = True


ROOT = Path(__file__).resolve().parents[2]


DIRS_TO_CREATE = [
    "outputs",
    "outputs/logs",
    "outputs/checkpoints",
    "outputs/analysis",
    "partitions",
    "partitions/hateful_memes",
    "partitions/crisismmd",
    "partitions/vqav2",
    "partitions/legacy_vqa_or_old",
    "docs",
    "scripts/maintenance",
]


COPY_ITEMS = [
    CopyItem("logging", "outputs/logs/logging", "Legacy logging dir backup", mode="all", optional=True),
    CopyItem("loggs", "outputs/logs/loggs", "Legacy log/output dir backup", mode="all", optional=True),
    CopyItem("logs", "outputs/logs/logs", "Optional logs dir backup", mode="all", optional=True),
    CopyItem("logs crisismmid-classfication-minicpm-0.5", "outputs/logs/logs crisismmid-classfication-minicpm-0.5", "Optional windows-style spaced logs dir", mode="all", optional=True),
    CopyItem("analysis", "outputs/analysis/current", "Analysis snapshot", mode="all", optional=True),
    CopyItem("partition-alpha0.1-clt10", "partitions/legacy_vqa_or_old/partition-alpha0.1-clt10", "Legacy partition snapshot", mode="all", optional=True),
    CopyItem("partition-alpha0.5-clt10", "partitions/legacy_vqa_or_old/partition-alpha0.5-clt10", "Legacy partition snapshot", mode="all", optional=True),
    CopyItem("partition-alpha1.0-clt10", "partitions/legacy_vqa_or_old/partition-alpha1.0-clt10", "Legacy partition snapshot", mode="all", optional=True),
    CopyItem("partition_vqav2_supercat_dirichlet_clients10", "partitions/vqav2/supercat_dirichlet_clients10", "Primary VQAv2 partition snapshot", mode="all", optional=True),
    CopyItem("crisismmd_datasplit_all", "partitions/crisismmd/datasplit_all", "CrisisMMD datasplit snapshot", mode="all", optional=True),
    CopyItem("hateful_memes/federated", "partitions/hateful_memes/federated", "Lightweight hateful_memes federated split copy (no image copy)", mode="lightweight_federated", optional=True),
]


UNKNOWN_SKIP_ITEMS = [
    "hateful_memes root directory: contains active dataset + images; keep in place.",
    "test.json root file: exact ownership/use may vary across workflows.",
    "FedMLLM legacy relative-path pipelines: not migrated in this light pass.",
]


ALLOWED_LIGHT_SUFFIX = {".json", ".jsonl", ".txt", ".md", ".csv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-risk migration helper for OpenFedMLLM")
    p.add_argument("--dry-run", action="store_true", help="Print actions without copying")
    p.add_argument(
        "--report",
        default="docs/MIGRATION_LOG.generated.json",
        help="Report path relative to repo root",
    )
    return p.parse_args()


def mkdirs(paths: Iterable[str], dry_run: bool, report: dict) -> None:
    for rel in paths:
        path = ROOT / rel
        report["created_dirs"].append(rel)
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path, dry_run: bool) -> None:
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_lightweight_federated(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        if p.suffix.lower() not in ALLOWED_LIGHT_SUFFIX:
            skipped += 1
            continue
        target = dst / rel
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        copied += 1
    return copied, skipped


def main() -> int:
    args = parse_args()
    dry_run = bool(args.dry_run)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")
    report = {
        "timestamp": now,
        "repo_root": str(ROOT),
        "dry_run": dry_run,
        "created_dirs": [],
        "copied": [],
        "missing_optional_sources": [],
        "skipped": [],
        "unknown_items": list(UNKNOWN_SKIP_ITEMS),
        "notes": [
            "No source files are deleted in this migration helper.",
            "No training/evaluation code path is modified by this script.",
        ],
    }

    mkdirs(DIRS_TO_CREATE, dry_run=dry_run, report=report)

    for item in COPY_ITEMS:
        src = ROOT / item.src
        dst = ROOT / item.dst
        if not src.exists():
            if item.optional:
                report["missing_optional_sources"].append(item.src)
                continue
            raise FileNotFoundError(f"Required source missing: {src}")

        if item.mode == "all":
            copy_tree(src=src, dst=dst, dry_run=dry_run)
            report["copied"].append(
                {
                    "src": item.src,
                    "dst": item.dst,
                    "mode": "all",
                    "reason": item.reason,
                }
            )
        elif item.mode == "lightweight_federated":
            copied, skipped = copy_lightweight_federated(src=src, dst=dst, dry_run=dry_run)
            report["copied"].append(
                {
                    "src": item.src,
                    "dst": item.dst,
                    "mode": "lightweight_federated",
                    "reason": item.reason,
                    "copied_files": copied,
                    "skipped_nonlight_files": skipped,
                }
            )
        else:
            report["skipped"].append(f"UNKNOWN mode for {item.src}: {item.mode}")

    report_path = ROOT / args.report
    if not dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
