#!/usr/bin/env python3
"""Compute Cross-Heterogeneity Interaction (CHI) from experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SETTING_TO_CELL = {
    "L0_M0": "A_00",
    "L1_M0": "A_L0",
    "L0_M1": "A_0M",
    "L1_M1": "A_LM",
}
CELL_ORDER = ["A_00", "A_L0", "A_0M", "A_LM"]
METRIC_CANDIDATES = [
    "final_accuracy",
    "accuracy",
    "best_accuracy",
    "acc",
    "normalized_exact_match",
    "vqa_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CHI summaries from experiment output dirs.")
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metric", default=None, help="Metric key to use; defaults to accuracy-like keys.")
    parser.add_argument("--fedavg_algorithm", default="FedAvg-LoRA")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(output_dir: Path) -> dict[str, Any] | None:
    for name in ("final_summary.json", "eval_metrics.json", "eval/eval_metrics.json"):
        path = output_dir / name
        if path.exists():
            payload = read_json(path)
            payload.setdefault("output_dir", str(output_dir))
            return payload
    return None


def normalize_algorithm(payload: dict[str, Any], output_dir: Path) -> str:
    for key in ("algorithm", "fed_alg", "mode"):
        value = payload.get(key)
        if value:
            return canonical_algorithm(str(value))
    text = output_dir.name.lower()
    if "fedchi" in text:
        return "FedCHI"
    if "fedprox" in text:
        return "FedProx-LoRA"
    if "fedavg" in text:
        return "FedAvg-LoRA"
    return "UNKNOWN"


def canonical_algorithm(value: str) -> str:
    raw = value.strip()
    lower = raw.lower().replace("_", "-")
    if "fedchi" in lower:
        if "no-consistency" in lower or "no-cons" in lower:
            return "FedCHI-no-consistency"
        if "no-hetagg" in lower or "no-het-agg" in lower:
            return "FedCHI-no-hetagg"
        if "no-modality" in lower:
            return "FedCHI-no-modality-lora"
        if "no-shared" in lower:
            return "FedCHI-no-shared-lora"
        return "FedCHI"
    if "fedprox" in lower:
        return "FedProx-LoRA"
    if "shared-lora-only" in lower or "shared-only" in lower:
        return "FedCHI-shared-lora-only"
    if "fedavg" in lower:
        return "FedAvg-LoRA"
    return raw


def normalize_setting(payload: dict[str, Any], output_dir: Path) -> str | None:
    for key in ("setting", "chi_setting"):
        value = payload.get(key)
        if value:
            return str(value).upper()
    label = payload.get("label_level")
    modality = payload.get("modality_level")
    if label and modality:
        return f"{label}_{modality}".upper()
    text = str(output_dir).upper()
    for setting in SETTING_TO_CELL:
        if setting in text:
            return setting
    return None


def metric_value(payload: dict[str, Any], metric: str | None) -> float | None:
    candidates = [metric] if metric else METRIC_CANDIDATES
    for key in candidates:
        if not key:
            continue
        value = payload.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    eval_metrics = payload.get("eval_metrics")
    if isinstance(eval_metrics, dict):
        for key in candidates:
            value = eval_metrics.get(key)
            if isinstance(value, (float, int)):
                return float(value)
    return None


def collect(input_dirs: list[str], metric: str | None) -> dict[str, dict[str, dict[str, Any]]]:
    table: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw_dir in input_dirs:
        output_dir = Path(raw_dir)
        payload = load_summary(output_dir)
        if payload is None:
            continue
        algorithm = normalize_algorithm(payload, output_dir)
        setting = normalize_setting(payload, output_dir)
        value = metric_value(payload, metric)
        if setting not in SETTING_TO_CELL or value is None:
            continue
        cell = SETTING_TO_CELL[setting]
        table[algorithm][cell] = {
            "value": float(value),
            "setting": setting,
            "output_dir": str(output_dir),
        }
    return table


def compute_row(
    algorithm: str,
    cells: dict[str, dict[str, Any]],
    fedavg_chi: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"algorithm": algorithm}
    values: dict[str, float | None] = {}
    for cell in CELL_ORDER:
        value = cells.get(cell, {}).get("value")
        values[cell] = float(value) if value is not None else None
        out[cell] = values[cell] if values[cell] is not None else "MISSING"

    if any(values[cell] is None for cell in CELL_ORDER):
        out.update({
            "D_L": "MISSING",
            "D_M": "MISSING",
            "D_LM": "MISSING",
            "CHI": "MISSING",
            "chi_reduction_relative_to_fedavg": "MISSING",
            "fedchi_reduces_chi_vs_fedavg": "MISSING",
        })
        return out

    a00 = values["A_00"]
    al0 = values["A_L0"]
    a0m = values["A_0M"]
    alm = values["A_LM"]
    assert a00 is not None and al0 is not None and a0m is not None and alm is not None
    d_l = a00 - al0
    d_m = a00 - a0m
    d_lm = a00 - alm
    chi = al0 + a0m - a00 - alm
    out.update({
        "D_L": d_l,
        "D_M": d_m,
        "D_LM": d_lm,
        "CHI": chi,
    })
    if fedavg_chi is None:
        out["chi_reduction_relative_to_fedavg"] = "MISSING"
        out["fedchi_reduces_chi_vs_fedavg"] = "MISSING"
    else:
        out["chi_reduction_relative_to_fedavg"] = fedavg_chi - chi
        out["fedchi_reduces_chi_vs_fedavg"] = bool(chi < fedavg_chi)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "algorithm",
        "A_00",
        "A_L0",
        "A_0M",
        "A_LM",
        "D_L",
        "D_M",
        "D_LM",
        "CHI",
        "chi_reduction_relative_to_fedavg",
        "fedchi_reduces_chi_vs_fedavg",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], fedavg_algorithm: str) -> None:
    headers = [
        "algorithm",
        "A_00",
        "A_L0",
        "A_0M",
        "A_LM",
        "D_L",
        "D_M",
        "D_LM",
        "CHI",
        "CHI reduction vs FedAvg",
        "FedCHI < FedAvg CHI",
    ]
    lines = [
        "# CHI Summary",
        "",
        f"FedAvg reference algorithm: `{fedavg_algorithm}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.get("algorithm"),
            row.get("A_00"),
            row.get("A_L0"),
            row.get("A_0M"),
            row.get("A_LM"),
            row.get("D_L"),
            row.get("D_M"),
            row.get("D_LM"),
            row.get("CHI"),
            row.get("chi_reduction_relative_to_fedavg"),
            row.get("fedchi_reduces_chi_vs_fedavg"),
        ]
        lines.append("| " + " | ".join(format_value(x) for x in values) + " |")
    lines.append("")
    lines.append("Primary criterion: `CHI_FedCHI < CHI_FedAvg-LoRA`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = collect(args.input_dirs, metric=args.metric)

    fedavg_cells = table.get(args.fedavg_algorithm, {})
    fedavg_row = compute_row(args.fedavg_algorithm, fedavg_cells, fedavg_chi=None)
    fedavg_chi = fedavg_row["CHI"] if isinstance(fedavg_row.get("CHI"), float) else None

    rows = []
    for algorithm in sorted(table):
        rows.append(compute_row(algorithm, table[algorithm], fedavg_chi=fedavg_chi))
    if args.fedavg_algorithm not in table:
        rows.insert(0, fedavg_row)

    write_csv(args.output_dir / "chi_summary.csv", rows)
    write_markdown(args.output_dir / "chi_summary.md", rows, fedavg_algorithm=args.fedavg_algorithm)
    with (args.output_dir / "chi_by_algorithm.json").open("w", encoding="utf-8") as f:
        json.dump({"algorithms": rows, "raw_cells": table}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote CHI summary -> {args.output_dir}")


if __name__ == "__main__":
    main()
