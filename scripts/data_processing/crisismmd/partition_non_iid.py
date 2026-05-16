#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from crisismmd.constants import DEFAULT_STANDARDIZED_DIRNAME, MANIFEST_FILENAME, STANDARD_SPLIT_FILES
    from crisismmd.dataset_utils import load_standardized_split, standardized_to_training_sample
    from crisismmd.io_utils import ensure_dir, load_manifest, write_json
    from crisismmd.stats import write_dataset_reports
else:
    from .constants import DEFAULT_STANDARDIZED_DIRNAME, MANIFEST_FILENAME, STANDARD_SPLIT_FILES
    from .dataset_utils import load_standardized_split, standardized_to_training_sample
    from .io_utils import ensure_dir, load_manifest, write_json
    from .stats import write_dataset_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Dirichlet Non-IID federated partitions from standardized CrisisMMD humanitarian data."
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default=DEFAULT_STANDARDIZED_DIRNAME,
        help="Standardized dataset directory produced by standardize_humanitarian.py",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output federated dataset directory.")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of federated clients.")
    parser.add_argument("--alpha", type=float, required=True, help="Dirichlet alpha for label-semantic Non-IID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--copy_eval_splits",
        action="store_true",
        help="Also convert and copy dev/test splits into the output directory.",
    )
    return parser.parse_args()


def dirichlet_partition(
    train_records: Sequence[Dict[str, Any]],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be > 0")
    if alpha <= 0:
        raise ValueError("--alpha must be > 0")

    rng = np.random.default_rng(seed)
    label_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        label_to_records[str(record["label"])].append(record)

    client_records: List[List[Dict[str, Any]]] = [[] for _ in range(num_clients)]
    for label in sorted(label_to_records.keys()):
        group = list(label_to_records[label])
        rng.shuffle(group)
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(group), probs)
        start = 0
        for client_id, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            end = start + count
            client_records[client_id].extend(group[start:end])
            start = end

    for client_id in range(num_clients):
        rng.shuffle(client_records[client_id])

    # Repair empty clients by moving one sample from the largest client.
    for client_id, records in enumerate(client_records):
        if records:
            continue
        donor_id = max(range(num_clients), key=lambda idx: len(client_records[idx]))
        if not client_records[donor_id]:
            raise RuntimeError("Partition repair failed: no samples available to repair empty client.")
        donor_records = client_records[donor_id]
        donor_counts = Counter(record["label"] for record in donor_records)
        dominant_label = max(donor_counts.items(), key=lambda item: (item[1], item[0]))[0]
        move_idx = next(idx for idx, record in enumerate(donor_records) if record["label"] == dominant_label)
        client_records[client_id].append(donor_records.pop(move_idx))

    return client_records


def convert_eval_split(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        standardized_to_training_sample(
            record=record,
            has_image=bool(record["modalities"]["image_available"]),
            has_text=bool(record["modalities"]["text_available"]),
            transformation_notes=["eval_export"],
            client_id=None,
        )
        for record in records
    ]


def main() -> int:
    args = parse_args()
    input_dataset = Path(args.input_dataset)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    manifest = load_manifest(input_dataset)
    if not str(manifest.get("dataset_type", "")).startswith("standardized"):
        raise ValueError(
            f"--input_dataset must point to a standardized humanitarian dataset, got {manifest.get('dataset_type')!r}"
        )

    train_records = load_standardized_split(input_dataset, "train")
    client_records = dirichlet_partition(
        train_records=train_records,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )

    for client_id, records in enumerate(client_records):
        payload = [
            standardized_to_training_sample(
                record=record,
                has_image=bool(record["modalities"]["image_available"]),
                has_text=bool(record["modalities"]["text_available"]),
                transformation_notes=[
                    f"dirichlet_non_iid(alpha={args.alpha}, seed={args.seed})",
                ],
                client_id=client_id,
            )
            for record in records
        ]
        write_json(output_dir / f"client_{client_id}.json", payload)

    copied_eval = {}
    if args.copy_eval_splits:
        for split in ("dev", "test"):
            eval_records = load_standardized_split(input_dataset, split)
            write_json(output_dir / STANDARD_SPLIT_FILES[split], convert_eval_split(eval_records))
            copied_eval[split] = STANDARD_SPLIT_FILES[split]

    out_manifest = {
        "dataset_type": "federated_humanitarian",
        "task": "humanitarian",
        "label_source": manifest.get("label_source"),
        "base_dataset": str(input_dataset).replace("\\", "/"),
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "seed": args.seed,
        "client_files": [f"client_{idx}.json" for idx in range(args.num_clients)],
        "eval_files": copied_eval,
        "transform_history": [
            {
                "name": "dirichlet_non_iid",
                "alpha": args.alpha,
                "seed": args.seed,
                "num_clients": args.num_clients,
                "comment": "Label-semantic Non-IID partition over train split using label_image-aligned supervision.",
            }
        ],
    }
    write_json(output_dir / MANIFEST_FILENAME, out_manifest)
    write_dataset_reports(output_dir)
    print(f"Federated Non-IID dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
