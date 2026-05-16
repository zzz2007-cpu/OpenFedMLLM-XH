#!/usr/bin/env python3
"""
Aggregate benchmark results from multiple experiment runs into a CSV table.

Usage:
    python mllmzoo/aggregate_results.py \\
        --results_dirs \\
            ./mllmzoo/output/fedavg \\
            ./mllmzoo/output/fedprox \\
            ./mllmzoo/output/scaffold \\
        --method_names FedAvg FedProx SCAFFOLD \\
        --output_csv   ./mllmzoo/output/benchmark_table.csv

Each results_dir must contain eval_metrics.json (written by eval_fed_model.py).

Outputs:
    benchmark_table.csv    — ready-to-paste table for a paper
    benchmark_table.md     — markdown version of the same table
"""

import argparse
import csv
import json
import os
import sys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate eval_metrics.json files into a benchmark table"
    )
    parser.add_argument(
        "--results_dirs", nargs="+", required=True,
        help="Directories containing eval_metrics.json (one per method/experiment)"
    )
    parser.add_argument(
        "--method_names", nargs="+", default=None,
        help="Display names for each results_dir (same order). "
             "Defaults to the directory basename."
    )
    parser.add_argument(
        "--output_csv", default="./mllmzoo/output/benchmark_table.csv",
        help="Path to write the output CSV"
    )
    parser.add_argument(
        "--also_read_per_round", action="store_true",
        help="If set, also read eval_metrics_per_round.jsonl and report best/last round"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRIC_COLUMNS = [
    ("accuracy",    "Accuracy"),
    ("f1_weighted", "F1 (Weighted)"),
    ("f1_macro",    "F1 (Macro)"),
    ("num_samples", "# Test Samples"),
]


def load_metrics(results_dir: str) -> dict:
    """Load scalar metrics from eval_metrics.json in a results directory."""
    path = os.path.join(results_dir, "eval_metrics.json")
    if not os.path.exists(path):
        print(f"  [WARNING] eval_metrics.json not found in: {results_dir}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_per_round_metrics(results_dir: str) -> list:
    """Load per-round metrics from eval_metrics_per_round.jsonl."""
    path = os.path.join(results_dir, "eval_metrics_per_round.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dirs = args.results_dirs
    names = args.method_names or [os.path.basename(d.rstrip("/\\")) for d in dirs]

    if len(dirs) != len(names):
        print("ERROR: --results_dirs and --method_names must have the same length.")
        sys.exit(1)

    rows = []
    for method_name, results_dir in zip(names, dirs):
        metrics = load_metrics(results_dir)
        row = {"Method": method_name}

        for key, col in _METRIC_COLUMNS:
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                row[col] = f"{val:.4f}"
            else:
                row[col] = str(val)

        if args.also_read_per_round:
            per_round = load_per_round_metrics(results_dir)
            if per_round:
                best = max(per_round, key=lambda r: r.get("accuracy", 0))
                last = per_round[-1]
                row["Best Round"] = str(best.get("round_idx", "?") + 1)
                row["Best Acc"]   = f"{best.get('accuracy', 0):.4f}"
                row["Last Acc"]   = f"{last.get('accuracy', 0):.4f}"

        rows.append(row)
        print(f"  Loaded: {method_name}  →  {metrics}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    if not rows:
        print("No results found.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[aggregate] CSV  → {args.output_csv}")

    # ── Write Markdown ────────────────────────────────────────────────────
    md_path = args.output_csv.replace(".csv", ".md")
    _write_markdown_table(rows, fieldnames, md_path)
    print(f"[aggregate] Markdown → {md_path}")

    # ── Print to console ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    _print_table(rows, fieldnames)
    print("=" * 70)


def _write_markdown_table(rows, fieldnames, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Results\n\n")
        # Header
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in fieldnames) + " |\n")
        f.write("\n> Generated by `mllmzoo/aggregate_results.py`\n")


def _print_table(rows, fieldnames):
    col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
                  for c in fieldnames}
    header = "  ".join(c.ljust(col_widths[c]) for c in fieldnames)
    print(header)
    print("  ".join("-" * col_widths[c] for c in fieldnames))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in fieldnames))


if __name__ == "__main__":
    main()
