#!/usr/bin/env python3
"""Generate CrisisMMD humanitarian federated partitions with Dirichlet label skew.

Output format follows the existing partition directory style:
  output_dir/
    client_0.json
    client_1.json
    ...
    client_{N-1}.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_LABEL_ORDER = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
    "other_relevant_information",
    "not_humanitarian",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Dirichlet-based federated split for humanitarian task.")
    parser.add_argument("--alpha", type=float, required=True, help="Dirichlet concentration parameter (>0).")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients.")
    parser.add_argument(
        "--input_tsv",
        type=str,
        default="crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv",
        help="Centralized train TSV with image_id and label_image columns.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output partition directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="partition-alpha0.5-clt10",
        help="Reference partition directory used as JSON template source.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output directory (client_*.json will be replaced).",
    )
    parser.add_argument(
        "--json_indent",
        type=int,
        default=2,
        help="Indent spaces for output JSON. Set negative value for compact one-line JSON.",
    )
    return parser.parse_args()


def normalize_label(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")


def choose_label_order(label_set: set[str]) -> List[str]:
    if set(DEFAULT_LABEL_ORDER) == label_set:
        return list(DEFAULT_LABEL_ORDER)
    return sorted(label_set)


def load_central_tsv(path: Path) -> Dict[str, str]:
    id_to_label: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = set(reader.fieldnames or [])
        required = {"image_id", "label_image"}
        if not required.issubset(cols):
            raise ValueError(f"TSV missing required columns {required}, got {reader.fieldnames}")
        for row in reader:
            sid = str(row["image_id"]).strip()
            lbl = normalize_label(str(row["label_image"]))
            if not sid:
                continue
            if sid in id_to_label and id_to_label[sid] != lbl:
                raise ValueError(f"Duplicate image_id with inconsistent labels: {sid}")
            id_to_label[sid] = lbl
    if not id_to_label:
        raise ValueError("No rows loaded from input_tsv.")
    return id_to_label


def load_reference_records(reference_dir: Path, num_clients: int) -> Dict[str, dict]:
    id_to_record: Dict[str, dict] = {}
    for cid in range(num_clients):
        fp = reference_dir / f"client_{cid}.json"
        if not fp.exists():
            raise FileNotFoundError(f"Missing reference client file: {fp}")
        records = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{fp} must be a JSON list.")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("id") or rec.get("image_id") or "").strip()
            if not sid:
                raise ValueError(f"Found record without id in {fp}")
            if sid in id_to_record:
                raise ValueError(f"Duplicate sample id in reference partition: {sid}")
            id_to_record[sid] = rec
    return id_to_record


def partition_by_dirichlet(
    id_to_label: Dict[str, str],
    label_order: Sequence[str],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[str]]:
    if alpha <= 0:
        raise ValueError("--alpha must be > 0.")
    rng = np.random.default_rng(seed)
    label_to_ids: Dict[str, List[str]] = {lbl: [] for lbl in label_order}
    for sid, lbl in id_to_label.items():
        if lbl not in label_to_ids:
            raise ValueError(f"Unknown label found in TSV: {lbl}, sample_id={sid}")
        label_to_ids[lbl].append(sid)

    client_ids: List[List[str]] = [[] for _ in range(num_clients)]
    for lbl in label_order:
        ids = label_to_ids[lbl]
        if not ids:
            continue
        ids = list(rng.permutation(ids))
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
        alloc = rng.multinomial(len(ids), probs)
        start = 0
        for cid, n in enumerate(alloc):
            end = start + int(n)
            if end > start:
                client_ids[cid].extend(ids[start:end])
            start = end

    for cid in range(num_clients):
        rng.shuffle(client_ids[cid])
    return client_ids


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.allclose(x.sum(), 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    numer = 2.0 * np.sum(np.arange(1, n + 1) * x_sorted)
    denom = n * np.sum(x_sorted)
    return float(numer / denom - (n + 1) / n)


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    if mean == 0:
        return 0.0
    return float(np.std(x, ddof=1) / mean)


def summarize_partition(
    client_ids: Sequence[Sequence[str]],
    id_to_label: Dict[str, str],
    label_order: Sequence[str],
) -> dict:
    label_to_idx = {l: i for i, l in enumerate(label_order)}
    n_clients = len(client_ids)
    n_labels = len(label_order)
    counts = np.zeros((n_clients, n_labels), dtype=np.int64)
    for cid, ids in enumerate(client_ids):
        for sid in ids:
            counts[cid, label_to_idx[id_to_label[sid]]] += 1

    global_counts = counts.sum(axis=0)
    global_prob = global_counts / global_counts.sum()
    client_sizes = counts.sum(axis=1)
    probs = np.divide(
        counts,
        client_sizes[:, None],
        out=np.zeros_like(counts, dtype=float),
        where=client_sizes[:, None] > 0,
    )
    kl_per_client = np.array([kl_divergence(probs[i], global_prob) for i in range(n_clients)], dtype=float)

    return {
        "total_samples": int(global_counts.sum()),
        "client_sizes": {f"client_{i}": int(client_sizes[i]) for i in range(n_clients)},
        "label_distribution_per_client": {
            f"client_{i}": {
                label_order[j]: {
                    "count": int(counts[i, j]),
                    "prob": float(probs[i, j]),
                }
                for j in range(n_labels)
            }
            for i in range(n_clients)
        },
        "global_distribution": {
            label_order[j]: {
                "count": int(global_counts[j]),
                "prob": float(global_prob[j]),
            }
            for j in range(n_labels)
        },
        "kl": {
            "per_client": {f"client_{i}": float(kl_per_client[i]) for i in range(n_clients)},
            "mean": float(np.mean(kl_per_client)),
            "min": float(np.min(kl_per_client)),
            "max": float(np.max(kl_per_client)),
        },
        "quantity_skew": {
            "cv": float(cv(client_sizes)),
            "gini": float(gini_coefficient(client_sizes)),
        },
    }


def write_partition(
    output_dir: Path,
    client_ids: Sequence[Sequence[str]],
    id_to_record: Dict[str, dict],
    overwrite: bool,
    json_indent: int,
) -> None:
    if output_dir.exists():
        if not overwrite:
            existing = list(output_dir.glob("client_*.json"))
            if existing:
                raise FileExistsError(
                    f"{output_dir} already has client_*.json files. Use --overwrite to replace them."
                )
    output_dir.mkdir(parents=True, exist_ok=True)

    for cid, ids in enumerate(client_ids):
        records = [id_to_record[sid] for sid in ids]
        out_fp = output_dir / f"client_{cid}.json"
        with out_fp.open("w", encoding="utf-8") as f:
            if json_indent >= 0:
                json.dump(records, f, ensure_ascii=False, indent=json_indent)
            else:
                json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")


def validate_exact_cover(client_ids: Sequence[Sequence[str]], expected_ids: set[str]) -> None:
    merged = [sid for ids in client_ids for sid in ids]
    merged_set = set(merged)
    if len(merged_set) != len(merged):
        raise ValueError("Duplicate sample ids detected across clients.")
    if merged_set != expected_ids:
        miss = sorted(expected_ids - merged_set)[:5]
        extra = sorted(merged_set - expected_ids)[:5]
        raise ValueError(
            "Partition does not exactly cover centralized train IDs. "
            f"missing_examples={miss}, extra_examples={extra}"
        )


def save_summary(
    output_dir: Path,
    summary: dict,
    alpha: float,
    num_clients: int,
    seed: int,
    json_indent: int,
    input_tsv: str,
    reference_dir: str,
) -> None:
    payload = {
        "meta": {
            "alpha": alpha,
            "num_clients": num_clients,
            "seed": seed,
            "json_indent": json_indent,
            "input_tsv": input_tsv,
            "reference_dir": reference_dir,
            "format": "client_*.json in output root (compatible with partition-alpha0.5-clt10)",
        },
        "summary": summary,
    }
    with (output_dir / "partition_stats.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with (output_dir / "partition_stats.txt").open("w", encoding="utf-8") as f:
        f.write(f"alpha={alpha}, num_clients={num_clients}, seed={seed}\n")
        f.write(f"total_samples={summary['total_samples']}\n")
        f.write(f"mean_kl={summary['kl']['mean']:.6f}\n")
        f.write(f"quantity_cv={summary['quantity_skew']['cv']:.6f}, quantity_gini={summary['quantity_skew']['gini']:.6f}\n")
        f.write("client_sizes:\n")
        for k, v in summary["client_sizes"].items():
            f.write(f"  {k}: {v}\n")


def main() -> None:
    args = parse_args()
    input_tsv = Path(args.input_tsv)
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)

    id_to_label = load_central_tsv(input_tsv)
    label_order = choose_label_order(set(id_to_label.values()))
    id_to_record = load_reference_records(reference_dir, args.num_clients)

    central_ids = set(id_to_label.keys())
    ref_ids = set(id_to_record.keys())
    if central_ids != ref_ids:
        miss = sorted(central_ids - ref_ids)[:5]
        extra = sorted(ref_ids - central_ids)[:5]
        raise ValueError(
            "Reference partition IDs do not match centralized train IDs. "
            f"missing_examples={miss}, extra_examples={extra}"
        )

    client_ids = partition_by_dirichlet(
        id_to_label=id_to_label,
        label_order=label_order,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    validate_exact_cover(client_ids, central_ids)
    write_partition(
        output_dir,
        client_ids,
        id_to_record,
        overwrite=args.overwrite,
        json_indent=args.json_indent,
    )
    summary = summarize_partition(client_ids, id_to_label, label_order)
    save_summary(
        output_dir=output_dir,
        summary=summary,
        alpha=args.alpha,
        num_clients=args.num_clients,
        seed=args.seed,
        json_indent=args.json_indent,
        input_tsv=str(input_tsv),
        reference_dir=str(reference_dir),
    )

    print(f"Generated partition at: {output_dir.resolve()}")
    print(f"alpha={args.alpha}, num_clients={args.num_clients}, seed={args.seed}")
    print(f"total_samples={summary['total_samples']}, mean_kl={summary['kl']['mean']:.6f}")
    print(
        "quantity_skew: "
        f"cv={summary['quantity_skew']['cv']:.6f}, gini={summary['quantity_skew']['gini']:.6f}"
    )
    print("client_sizes:", summary["client_sizes"])


if __name__ == "__main__":
    main()
