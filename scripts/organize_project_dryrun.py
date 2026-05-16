#!/usr/bin/env python3
"""Print a proposed non-destructive project reorganization plan.

This script is intentionally dry-run only. It does not move, copy, rename, or
delete anything. Review PROJECT_STRUCTURE.md before turning any item here into
a real migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MovePlan:
    source: str
    target: str
    reason: str
    risk: str = "medium"


REPO_ROOT = Path(__file__).resolve().parents[1]


MOVE_PLANS = [
    MovePlan(
        source="crisismmd_datasplit_all",
        target="data/crisis-mmd/raw_splits/crisismmd_datasplit_all",
        reason="CrisisMMD original TSV split files; belongs with raw/processed data.",
        risk="high: several scripts and legacy FedMLLM evaluators reference this path directly.",
    ),
    MovePlan(
        source="VQAv2",
        target="data/vqav2/raw",
        reason="VQAv2 questions, annotations, and image folders; belongs under data.",
        risk="high: VQA configs and partition scripts hard-code ./VQAv2.",
    ),
    MovePlan(
        source="hateful_memes",
        target="data/hateful_memes/raw_and_partitions",
        reason="Hateful Memes raw JSONL, images, and generated federated partitions.",
        risk="high: open tabs, build scripts, and possible experiment configs use ./hateful_memes.",
    ),
    MovePlan(
        source="partition-alpha0.1-clt10",
        target="partitions/crisis-mmd/partition-alpha0.1-clt10",
        reason="CrisisMMD federated client shards for alpha=0.1.",
        risk="high: analysis defaults and cleanup examples reference partition-alpha*."),
    MovePlan(
        source="partition-alpha0.5-clt10",
        target="partitions/crisis-mmd/partition-alpha0.5-clt10",
        reason="CrisisMMD federated client shards for alpha=0.5.",
        risk="high: analysis defaults and cleanup examples reference partition-alpha*."),
    MovePlan(
        source="partition-alpha1.0-clt10",
        target="partitions/crisis-mmd/partition-alpha1.0-clt10",
        reason="CrisisMMD federated client shards for alpha=1.0.",
        risk="high: training configs may need data_path updates."),
    MovePlan(
        source="partition_vqav2_supercat_dirichlet_clients10",
        target="partitions/vqav2/supercat_dirichlet_clients10",
        reason="VQAv2 Supercategory-Dirichlet federated partition output.",
        risk="high: VQA configs hard-code ./partition_vqav2_supercat_dirichlet_clients10.",
    ),
    MovePlan(
        source="logging",
        target="outputs/legacy_logging",
        reason="Legacy FL/LLM log output directory.",
        risk="medium: older zoo configs use ./logging paths.",
    ),
    MovePlan(
        source="loggs",
        target="outputs/loggs",
        reason="Experiment log/output directory with nonstandard spelling.",
        risk="medium: exact writer is unclear; keep until references are audited.",
    ),
    MovePlan(
        source="mllmzoo/output",
        target="outputs/mllmzoo",
        reason="Model checkpoints, rank logs, predictions, and evaluation metrics.",
        risk="high: active training configs write to ./mllmzoo/output/...",
    ),
    MovePlan(
        source="partition_vqa_supercat_dirichlet.py",
        target="scripts/partitioning/partition_vqa_supercat_dirichlet.py",
        reason="Standalone VQAv2 partitioning utility; scripts/partitioning would group similar tools.",
        risk="medium: direct command examples or imports may reference repo-root script.",
    ),
    MovePlan(
        source="getdata.py",
        target="scripts/data_download/get_hateful_memes_hf.py",
        reason="Small Hateful Memes HuggingFace download helper.",
        risk="low: appears standalone, but keep root copy until confirmed.",
    ),
    MovePlan(
        source="generate_dummy_data.py",
        target="scripts/dev/generate_dummy_data.py",
        reason="Development/testing data helper.",
        risk="low: no main training path observed.",
    ),
    MovePlan(
        source="test.json",
        target="data/crisis-mmd/minicpmv_data/test.json",
        reason="Looks like CrisisMMD/MiniCPM-V evaluation JSON, but current purpose needs confirmation.",
        risk="high: root-level test.json is referenced by cleanup examples and may be manually used.",
    ),
    MovePlan(
        source="FedMLLM",
        target="legacy/FedMLLM",
        reason="Upstream/older FedMLLM code with separate data generation and evaluation scripts.",
        risk="high: still useful for reproducing legacy experiments; moving affects relative paths.",
    ),
    MovePlan(
        source="zoo",
        target="legacy/zoo_llm",
        reason="Older LLM experiment configs distinct from current mllmzoo MLLM configs.",
        risk="high: setup.py packages zoo and README references zoo examples.",
    ),
    MovePlan(
        source="flzoo",
        target="legacy/flzoo_classic",
        reason="Classic FL vision/text configs not central to current MLLM experiments.",
        risk="medium: setup.py packages flzoo; keep until import impact is checked.",
    ),
]


KEEP_IN_PLACE = [
    ("fling", "Core federated learning framework package; imported by setup.py."),
    ("fling_llm", "Core LLM federated package; imported by setup.py and older configs."),
    ("fling_mllm", "Current multimodal federated training/evaluation package."),
    ("mllmzoo", "Current model zoo, registry, and MLLM experiment configs."),
    ("scripts", "Current data processing, run wrappers, analysis helpers, and this dry-run script."),
    ("analysis", "Analysis scripts and generated reports; already grouped reasonably."),
    ("requirements.txt", "Base dependency file."),
    ("requirements-qwen2vl.txt", "Qwen2-VL dependency file."),
    ("setup.py", "Packaging and console-script entry points."),
    ("README.md", "Top-level project documentation."),
    ("LICENSE", "Project license."),
]


def format_status(path: Path) -> str:
    if path.exists():
        kind = "dir" if path.is_dir() else "file"
        return f"exists ({kind})"
    return "missing"


def main() -> int:
    print("OpenFedMLLM organization dry-run")
    print(f"Repo root: {REPO_ROOT}")
    print()
    print("No filesystem changes will be made.")
    print()

    print("Keep in place for now:")
    for source, reason in KEEP_IN_PLACE:
        print(f"  KEEP  {source:<28} [{format_status(REPO_ROOT / source)}]")
        print(f"        reason: {reason}")
    print()

    print("Proposed future moves:")
    for plan in MOVE_PLANS:
        src = REPO_ROOT / plan.source
        dst = REPO_ROOT / plan.target
        print(f"  MOVE  {plan.source} -> {plan.target}")
        print(f"        source: {format_status(src)}")
        print(f"        target currently: {format_status(dst)}")
        print(f"        reason: {plan.reason}")
        print(f"        risk: {plan.risk}")
        print()

    print("Dry-run complete. Review hard-coded paths before executing any real migration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
