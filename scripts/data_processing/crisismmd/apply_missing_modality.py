#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from crisismmd.constants import MANIFEST_FILENAME, STANDARD_SPLIT_FILES, SUPPORTED_MISSING_STRATEGIES
    from crisismmd.dataset_utils import sample_to_base_record, standardized_to_training_sample
    from crisismmd.io_utils import copy_optional_files, ensure_dir, list_client_files, load_manifest, read_json, write_json
    from crisismmd.stats import write_dataset_reports
else:
    from .constants import MANIFEST_FILENAME, STANDARD_SPLIT_FILES, SUPPORTED_MISSING_STRATEGIES
    from .dataset_utils import sample_to_base_record, standardized_to_training_sample
    from .io_utils import copy_optional_files, ensure_dir, list_client_files, load_manifest, read_json, write_json
    from .stats import write_dataset_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply missing-modality transformation on an existing federated CrisisMMD humanitarian dataset."
    )
    parser.add_argument("--input_dataset_dir", type=str, required=True, help="Existing federated dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for missing-modality dataset.")
    parser.add_argument("--missing_rate", type=float, required=True, help="Missing ratio in [0, 1].")
    parser.add_argument(
        "--missing_strategy",
        type=str,
        required=True,
        choices=SUPPORTED_MISSING_STRATEGIES,
        help="Missing strategy: image / text / random",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _choose_indices(eligible_indices: Sequence[int], rate: float, rng: np.random.Generator) -> List[int]:
    if rate <= 0 or not eligible_indices:
        return []
    k = int(round(len(eligible_indices) * rate))
    k = min(max(k, 0), len(eligible_indices))
    if k == 0:
        return []
    return sorted(rng.choice(np.asarray(list(eligible_indices), dtype=np.int64), size=k, replace=False).tolist())


def transform_client_samples(
    samples: List[Dict[str, Any]],
    missing_rate: float,
    missing_strategy: str,
    seed: int,
    client_id: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + client_id)
    base_records = [sample_to_base_record(sample) for sample in samples]

    if missing_strategy == "image":
        eligible = [idx for idx, record in enumerate(base_records) if record["modalities"]["image_available"]]
        selected = set(_choose_indices(eligible, missing_rate, rng))
        drop_targets = {idx: "image" for idx in selected}
    elif missing_strategy == "text":
        eligible = [idx for idx, record in enumerate(base_records) if record["modalities"]["text_available"]]
        selected = set(_choose_indices(eligible, missing_rate, rng))
        drop_targets = {idx: "text" for idx in selected}
    else:
        eligible = [
            idx
            for idx, record in enumerate(base_records)
            if record["modalities"]["image_available"] and record["modalities"]["text_available"]
        ]
        selected = set(_choose_indices(eligible, missing_rate, rng))
        drop_targets = {idx: rng.choice(["image", "text"]).item() for idx in selected}

    transformed: List[Dict[str, Any]] = []
    for idx, record in enumerate(base_records):
        has_image = bool(record["modalities"]["image_available"])
        has_text = bool(record["modalities"]["text_available"])
        target = drop_targets.get(idx)
        if target == "image":
            has_image = False
        elif target == "text":
            has_text = False

        note = f"missing(strategy={missing_strategy}, rate={missing_rate}, seed={seed}, client={client_id})"
        transformed.append(
            standardized_to_training_sample(
                record=record,
                has_image=has_image,
                has_text=has_text,
                transformation_notes=list(record.get("transformations", [])) + [note],
                client_id=client_id,
            )
        )
    return transformed


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.missing_rate <= 1.0:
        raise ValueError("--missing_rate must be in [0, 1]")

    input_dir = Path(args.input_dataset_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    manifest = load_manifest(input_dir)
    client_files = list_client_files(input_dir)
    if not client_files:
        raise ValueError(f"No client shards found in {input_dir}")

    for client_file in client_files:
        client_id = int(client_file.stem.split("_")[-1])
        samples = read_json(client_file)
        transformed = transform_client_samples(
            samples=samples,
            missing_rate=args.missing_rate,
            missing_strategy=args.missing_strategy,
            seed=args.seed,
            client_id=client_id,
        )
        write_json(output_dir / client_file.name, transformed)

    eval_files = copy_optional_files(input_dir, output_dir, [STANDARD_SPLIT_FILES["dev"], STANDARD_SPLIT_FILES["test"]])

    new_manifest = dict(manifest)
    new_manifest["base_dataset"] = str(input_dir).replace("\\", "/")
    new_manifest["dataset_type"] = "federated_humanitarian"
    new_manifest["eval_files"] = {split: name for split, name in (("dev", STANDARD_SPLIT_FILES["dev"]), ("test", STANDARD_SPLIT_FILES["test"])) if name in eval_files}
    transform_history = list(new_manifest.get("transform_history", []))
    transform_history.append(
        {
            "name": "missing_modality",
            "missing_rate": args.missing_rate,
            "missing_strategy": args.missing_strategy,
            "seed": args.seed,
            "comment": "Samples are retained; modalities are masked in prompt/image field instead of deleting records.",
        }
    )
    new_manifest["transform_history"] = transform_history
    write_json(output_dir / MANIFEST_FILENAME, new_manifest)
    write_dataset_reports(output_dir)
    print(f"Missing-modality dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

