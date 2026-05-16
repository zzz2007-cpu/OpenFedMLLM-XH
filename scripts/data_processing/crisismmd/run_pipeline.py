#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from crisismmd.apply_cross_modality import apply_cross_to_client, assign_client_patterns
    from crisismmd.apply_missing_modality import transform_client_samples
    from crisismmd.constants import (
        DEFAULT_IMAGE_PREFIX,
        DEFAULT_INPUT_ROOT,
        DEFAULT_LABEL_SOURCE,
        DEFAULT_STANDARDIZED_DIRNAME,
        MANIFEST_FILENAME,
        STANDARD_SPLIT_FILES,
    )
    from crisismmd.dataset_utils import (
        read_standardized_tsv,
        standardized_to_training_sample,
    )
    from crisismmd.io_utils import ensure_dir, list_client_files, load_manifest, read_json, write_json
    from crisismmd.partition_non_iid import convert_eval_split, dirichlet_partition
    from crisismmd.stats import write_dataset_reports
else:
    from .apply_cross_modality import apply_cross_to_client, assign_client_patterns
    from .apply_missing_modality import transform_client_samples
    from .constants import (
        DEFAULT_IMAGE_PREFIX,
        DEFAULT_INPUT_ROOT,
        DEFAULT_LABEL_SOURCE,
        DEFAULT_STANDARDIZED_DIRNAME,
        MANIFEST_FILENAME,
        STANDARD_SPLIT_FILES,
    )
    from .dataset_utils import read_standardized_tsv, standardized_to_training_sample
    from .io_utils import ensure_dir, list_client_files, load_manifest, read_json, write_json
    from .partition_non_iid import convert_eval_split, dirichlet_partition
    from .stats import write_dataset_reports


def parse_float_list(raw: str) -> List[float]:
    values = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot pipeline: standardize humanitarian TSV -> Dirichlet Non-IID -> optional missing/cross."
    )
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT, help="Directory containing humanitarian TSV files.")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for generated datasets.")
    parser.add_argument("--standardized_dir", type=str, default="", help="Optional explicit standardized dataset directory.")
    parser.add_argument("--image_prefix", type=str, default=DEFAULT_IMAGE_PREFIX, help="Relative image prefix stored in standardized records.")
    parser.add_argument("--label_source", type=str, default=DEFAULT_LABEL_SOURCE, choices=("label", "label_text", "label_image"))
    parser.add_argument("--num_clients", type=int, required=True, help="Number of federated clients.")
    parser.add_argument("--alphas", type=str, required=True, help="Comma-separated Dirichlet alpha list, e.g. 0.1,0.5,1.0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--copy_eval_splits", action="store_true", help="Copy dev/test in training JSON format.")
    parser.add_argument("--missing_rates", type=str, default="", help="Optional comma-separated missing-rate list.")
    parser.add_argument("--missing_strategy", type=str, default="random", choices=("image", "text", "random"))
    parser.add_argument("--cross_rates", type=str, default="", help="Optional comma-separated cross-rate list.")
    parser.add_argument("--cross_strategy", type=str, default="client_partition", choices=("client_partition", "client_probability"))
    return parser.parse_args()


def standardize_once(input_root: Path, output_dir: Path, image_prefix: str, label_source: str) -> None:
    ensure_dir(output_dir)
    split_files: Dict[str, str] = {}
    source_files: Dict[str, str] = {}
    for split in ("train", "dev", "test"):
        tsv_path = input_root / f"task_humanitarian_text_img_{split}.tsv"
        records = read_standardized_tsv(
            tsv_path=tsv_path,
            split=split,
            image_prefix=image_prefix,
            label_source=label_source,
        )
        split_name = STANDARD_SPLIT_FILES[split]
        write_json(output_dir / split_name, records)
        split_files[split] = split_name
        source_files[split] = str(tsv_path).replace("\\", "/")

    write_json(
        output_dir / MANIFEST_FILENAME,
        {
            "dataset_type": "standardized_humanitarian",
            "task": "humanitarian",
            "label_source": label_source,
            "image_prefix": image_prefix,
            "input_root": str(input_root).replace("\\", "/"),
            "source_files": source_files,
            "split_files": split_files,
        },
    )
    write_dataset_reports(output_dir)


def build_partition_dataset(standardized_dir: Path, output_dir: Path, alpha: float, num_clients: int, seed: int, copy_eval_splits: bool) -> None:
    ensure_dir(output_dir)
    manifest = load_manifest(standardized_dir)
    train_records = read_json(standardized_dir / STANDARD_SPLIT_FILES["train"])
    client_records = dirichlet_partition(train_records=train_records, num_clients=num_clients, alpha=alpha, seed=seed)

    for client_id, records in enumerate(client_records):
        payload = [
            standardized_to_training_sample(
                record=record,
                has_image=bool(record["modalities"]["image_available"]),
                has_text=bool(record["modalities"]["text_available"]),
                transformation_notes=[f"dirichlet_non_iid(alpha={alpha}, seed={seed})"],
                client_id=client_id,
            )
            for record in records
        ]
        write_json(output_dir / f"client_{client_id}.json", payload)

    eval_files = {}
    if copy_eval_splits:
        for split in ("dev", "test"):
            eval_records = read_json(standardized_dir / STANDARD_SPLIT_FILES[split])
            write_json(output_dir / STANDARD_SPLIT_FILES[split], convert_eval_split(eval_records))
            eval_files[split] = STANDARD_SPLIT_FILES[split]

    write_json(
        output_dir / MANIFEST_FILENAME,
        {
            "dataset_type": "federated_humanitarian",
            "task": "humanitarian",
            "label_source": manifest.get("label_source"),
            "base_dataset": str(standardized_dir).replace("\\", "/"),
            "num_clients": num_clients,
            "alpha": alpha,
            "seed": seed,
            "client_files": [f"client_{idx}.json" for idx in range(num_clients)],
            "eval_files": eval_files,
            "transform_history": [
                {
                    "name": "dirichlet_non_iid",
                    "alpha": alpha,
                    "seed": seed,
                    "num_clients": num_clients,
                }
            ],
        },
    )
    write_dataset_reports(output_dir)


def build_missing_dataset(input_dir: Path, output_dir: Path, missing_rate: float, missing_strategy: str, seed: int) -> None:
    ensure_dir(output_dir)
    manifest = load_manifest(input_dir)
    for client_file in list_client_files(input_dir):
        client_id = int(client_file.stem.split("_")[-1])
        samples = read_json(client_file)
        transformed = transform_client_samples(
            samples=samples,
            missing_rate=missing_rate,
            missing_strategy=missing_strategy,
            seed=seed,
            client_id=client_id,
        )
        write_json(output_dir / client_file.name, transformed)

    eval_files = {}
    for split in ("dev", "test"):
        src = input_dir / STANDARD_SPLIT_FILES[split]
        if src.exists():
            write_json(output_dir / STANDARD_SPLIT_FILES[split], read_json(src))
            eval_files[split] = STANDARD_SPLIT_FILES[split]

    new_manifest = dict(manifest)
    history = list(new_manifest.get("transform_history", []))
    history.append(
        {
            "name": "missing_modality",
            "missing_rate": missing_rate,
            "missing_strategy": missing_strategy,
            "seed": seed,
        }
    )
    new_manifest["base_dataset"] = str(input_dir).replace("\\", "/")
    new_manifest["eval_files"] = eval_files
    new_manifest["transform_history"] = history
    write_json(output_dir / MANIFEST_FILENAME, new_manifest)
    write_dataset_reports(output_dir)


def build_cross_dataset(input_dir: Path, output_dir: Path, cross_rate: float, cross_strategy: str, seed: int) -> None:
    ensure_dir(output_dir)
    manifest = load_manifest(input_dir)
    client_files = list_client_files(input_dir)
    pattern_map, archetype_map = assign_client_patterns(
        num_clients=len(client_files),
        cross_rate=cross_rate,
        cross_strategy=cross_strategy,
        seed=seed,
    )

    client_patterns = {}
    for client_file in client_files:
        client_id = int(client_file.stem.split("_")[-1])
        transformed = apply_cross_to_client(
            samples=read_json(client_file),
            client_id=client_id,
            cross_rate=cross_rate,
            cross_strategy=cross_strategy,
            client_pattern=pattern_map[client_id],
            seed=seed,
        )
        write_json(output_dir / client_file.name, transformed)
        client_patterns[f"client_{client_id}"] = {
            "pattern": pattern_map[client_id],
            "archetype": archetype_map[client_id],
        }

    eval_files = {}
    for split in ("dev", "test"):
        src = input_dir / STANDARD_SPLIT_FILES[split]
        if src.exists():
            write_json(output_dir / STANDARD_SPLIT_FILES[split], read_json(src))
            eval_files[split] = STANDARD_SPLIT_FILES[split]

    new_manifest = dict(manifest)
    history = list(new_manifest.get("transform_history", []))
    history.append(
        {
            "name": "cross_modality",
            "cross_rate": cross_rate,
            "cross_strategy": cross_strategy,
            "seed": seed,
        }
    )
    new_manifest["base_dataset"] = str(input_dir).replace("\\", "/")
    new_manifest["eval_files"] = eval_files
    new_manifest["client_patterns"] = client_patterns
    new_manifest["transform_history"] = history
    write_json(output_dir / MANIFEST_FILENAME, new_manifest)
    write_dataset_reports(output_dir)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    standardized_dir = Path(args.standardized_dir) if args.standardized_dir else output_root / DEFAULT_STANDARDIZED_DIRNAME
    if not (standardized_dir / MANIFEST_FILENAME).exists():
        standardize_once(
            input_root=Path(args.input_root),
            output_dir=standardized_dir,
            image_prefix=args.image_prefix,
            label_source=args.label_source,
        )

    missing_rates = parse_float_list(args.missing_rates)
    cross_rates = parse_float_list(args.cross_rates)
    alphas = parse_float_list(args.alphas)

    for alpha in alphas:
        non_iid_dir = output_root / f"humanitarian_non_iid_alpha{alpha}_clients{args.num_clients}"
        build_partition_dataset(
            standardized_dir=standardized_dir,
            output_dir=non_iid_dir,
            alpha=alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            copy_eval_splits=args.copy_eval_splits,
        )

        for missing_rate in missing_rates:
            missing_dir = output_root / (
                f"humanitarian_non_iid_alpha{alpha}_clients{args.num_clients}"
                f"_missing{int(round(missing_rate * 100)):02d}_{args.missing_strategy}"
            )
            build_missing_dataset(
                input_dir=non_iid_dir,
                output_dir=missing_dir,
                missing_rate=missing_rate,
                missing_strategy=args.missing_strategy,
                seed=args.seed,
            )

        for cross_rate in cross_rates:
            cross_dir = output_root / (
                f"humanitarian_non_iid_alpha{alpha}_clients{args.num_clients}"
                f"_cross{int(round(cross_rate * 100)):02d}_{args.cross_strategy}"
            )
            build_cross_dataset(
                input_dir=non_iid_dir,
                output_dir=cross_dir,
                cross_rate=cross_rate,
                cross_strategy=args.cross_strategy,
                seed=args.seed,
            )

        for missing_rate in missing_rates:
            missing_dir = output_root / (
                f"humanitarian_non_iid_alpha{alpha}_clients{args.num_clients}"
                f"_missing{int(round(missing_rate * 100)):02d}_{args.missing_strategy}"
            )
            for cross_rate in cross_rates:
                combo_dir = output_root / (
                    f"humanitarian_non_iid_alpha{alpha}_clients{args.num_clients}"
                    f"_missing{int(round(missing_rate * 100)):02d}_{args.missing_strategy}"
                    f"_cross{int(round(cross_rate * 100)):02d}_{args.cross_strategy}"
                )
                build_cross_dataset(
                    input_dir=missing_dir,
                    output_dir=combo_dir,
                    cross_rate=cross_rate,
                    cross_strategy=args.cross_strategy,
                    seed=args.seed,
                )

    print(f"Pipeline outputs written under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
