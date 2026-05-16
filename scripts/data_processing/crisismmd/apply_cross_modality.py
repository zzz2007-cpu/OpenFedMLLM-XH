#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from crisismmd.constants import MANIFEST_FILENAME, STANDARD_SPLIT_FILES, SUPPORTED_CROSS_STRATEGIES
    from crisismmd.dataset_utils import sample_to_base_record, standardized_to_training_sample
    from crisismmd.io_utils import copy_optional_files, ensure_dir, list_client_files, load_manifest, read_json, write_json
    from crisismmd.stats import write_dataset_reports
else:
    from .constants import MANIFEST_FILENAME, STANDARD_SPLIT_FILES, SUPPORTED_CROSS_STRATEGIES
    from .dataset_utils import sample_to_base_record, standardized_to_training_sample
    from .io_utils import copy_optional_files, ensure_dir, list_client_files, load_manifest, read_json, write_json
    from .stats import write_dataset_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply client-level modal cross transformation on an existing federated CrisisMMD humanitarian dataset."
    )
    parser.add_argument("--input_dataset_dir", type=str, required=True, help="Existing federated dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cross-modality dataset.")
    parser.add_argument("--cross_rate", type=float, required=True, help="Cross intensity in [0, 1].")
    parser.add_argument(
        "--cross_strategy",
        type=str,
        required=True,
        choices=SUPPORTED_CROSS_STRATEGIES,
        help="Cross strategy: client_partition / client_probability",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def assign_client_patterns(
    num_clients: int,
    cross_rate: float,
    cross_strategy: str,
    seed: int,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    rng = np.random.default_rng(seed)
    client_ids = np.arange(num_clients)
    rng.shuffle(client_ids)

    pattern_map: Dict[int, str] = {}
    archetype_map: Dict[int, str] = {}

    if cross_strategy == "client_partition":
        # Reference FedMLLM idea, but generalized:
        # cross_rate controls the fraction of clients that become hard single-modality clients.
        # The selected clients are evenly divided into image_only and text_only;
        # remaining clients stay multimodal.
        num_single = int(round(num_clients * cross_rate))
        num_single = min(max(num_single, 0), num_clients)
        image_only_count = (num_single + 1) // 2
        text_only_count = num_single // 2
        for idx, client_id in enumerate(client_ids.tolist()):
            if idx < image_only_count:
                pattern_map[client_id] = "image_only"
            elif idx < image_only_count + text_only_count:
                pattern_map[client_id] = "text_only"
            else:
                pattern_map[client_id] = "multimodal"
            archetype_map[client_id] = pattern_map[client_id]
        return pattern_map, archetype_map

    # Softer variant:
    # clients are assigned to multimodal / image-dominant / text-dominant archetypes;
    # cross_rate controls the probability of dropping the non-dominant modality within
    # the dominant clients, not a full hard assignment.
    num_biased = int(round(num_clients * cross_rate))
    num_biased = min(max(num_biased, 0), num_clients)
    image_dom = (num_biased + 1) // 2
    text_dom = num_biased // 2
    for idx, client_id in enumerate(client_ids.tolist()):
        if idx < image_dom:
            archetype_map[client_id] = "image_dominant"
        elif idx < image_dom + text_dom:
            archetype_map[client_id] = "text_dominant"
        else:
            archetype_map[client_id] = "multimodal"
        pattern_map[client_id] = archetype_map[client_id]
    return pattern_map, archetype_map


def apply_cross_to_client(
    samples: List[Dict[str, Any]],
    client_id: int,
    cross_rate: float,
    cross_strategy: str,
    client_pattern: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + client_id * 97)
    base_records = [sample_to_base_record(sample) for sample in samples]
    note = (
        f"cross(strategy={cross_strategy}, rate={cross_rate}, seed={seed}, "
        f"client={client_id}, pattern={client_pattern})"
    )
    transformed: List[Dict[str, Any]] = []

    for record in base_records:
        has_image = bool(record["modalities"]["image_available"])
        has_text = bool(record["modalities"]["text_available"])

        if cross_strategy == "client_partition":
            allow_image = client_pattern in {"multimodal", "image_only"}
            allow_text = client_pattern in {"multimodal", "text_only"}
            has_image = has_image and allow_image
            has_text = has_text and allow_text
        else:
            if client_pattern == "image_dominant" and has_text:
                has_text = bool(rng.random() >= cross_rate)
            elif client_pattern == "text_dominant" and has_image:
                has_image = bool(rng.random() >= cross_rate)

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
    if not 0.0 <= args.cross_rate <= 1.0:
        raise ValueError("--cross_rate must be in [0, 1]")

    input_dir = Path(args.input_dataset_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    manifest = load_manifest(input_dir)
    client_files = list_client_files(input_dir)
    num_clients = len(client_files)
    if num_clients == 0:
        raise ValueError(f"No client shards found in {input_dir}")

    pattern_map, archetype_map = assign_client_patterns(
        num_clients=num_clients,
        cross_rate=args.cross_rate,
        cross_strategy=args.cross_strategy,
        seed=args.seed,
    )

    client_pattern_meta = {}
    for client_file in client_files:
        client_id = int(client_file.stem.split("_")[-1])
        samples = read_json(client_file)
        transformed = apply_cross_to_client(
            samples=samples,
            client_id=client_id,
            cross_rate=args.cross_rate,
            cross_strategy=args.cross_strategy,
            client_pattern=pattern_map[client_id],
            seed=args.seed,
        )
        write_json(output_dir / client_file.name, transformed)
        client_pattern_meta[f"client_{client_id}"] = {
            "pattern": pattern_map[client_id],
            "archetype": archetype_map[client_id],
        }

    eval_files = copy_optional_files(input_dir, output_dir, [STANDARD_SPLIT_FILES["dev"], STANDARD_SPLIT_FILES["test"]])

    new_manifest = dict(manifest)
    new_manifest["base_dataset"] = str(input_dir).replace("\\", "/")
    new_manifest["dataset_type"] = "federated_humanitarian"
    new_manifest["eval_files"] = {split: name for split, name in (("dev", STANDARD_SPLIT_FILES["dev"]), ("test", STANDARD_SPLIT_FILES["test"])) if name in eval_files}
    new_manifest["client_patterns"] = client_pattern_meta
    transform_history = list(new_manifest.get("transform_history", []))
    transform_history.append(
        {
            "name": "cross_modality",
            "cross_rate": args.cross_rate,
            "cross_strategy": args.cross_strategy,
            "seed": args.seed,
            "comment": (
                "Cross follows client-level modality heterogeneity. "
                "The implementation is a refactored, reusable variant inspired by FedMLLM's mix/cross setup."
            ),
        }
    )
    new_manifest["transform_history"] = transform_history
    write_json(output_dir / MANIFEST_FILENAME, new_manifest)
    write_dataset_reports(output_dir)
    print(f"Cross-modality dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

