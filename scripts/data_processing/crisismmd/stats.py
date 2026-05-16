from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .constants import HUMANITARIAN_LABELS, MANIFEST_FILENAME, STATS_FILENAME, VALIDATION_FILENAME
from .dataset_utils import label_histogram, load_client_samples, sample_to_base_record
from .io_utils import list_client_files, load_manifest, read_json, repo_root, write_json


def _resolve_image_candidates(image_value: Any, dataset_dir: Path) -> List[Path]:
    if not isinstance(image_value, str) or not image_value.strip():
        return []

    image_path = image_value.strip()
    candidates: List[Path] = []
    p = Path(image_path)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((dataset_dir / image_path).absolute())
        candidates.append((repo_root() / image_path).absolute())
    if "data_image" in image_path:
        suffix = image_path.split("data_image", 1)[-1].lstrip("/\\")
        candidates.append((repo_root() / "data" / "crisis-mmd" / "raw_data" / "data_image" / suffix).absolute())
    dedup: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            dedup.append(candidate)
            seen.add(key)
    return dedup


def _client_stats(
    client_name: str,
    samples: List[Dict[str, Any]],
    dataset_dir: Path,
    path_checks: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    base_records = [sample_to_base_record(sample) for sample in samples]
    label_counts = label_histogram(base_records)
    pattern_counter = Counter()
    invalid_paths: List[str] = []
    missing_fields = 0

    image_available = 0
    text_available = 0

    for idx, sample in enumerate(samples):
        record = base_records[idx]
        modalities = record["modalities"]
        if modalities["image_available"]:
            image_available += 1
        if modalities["text_available"]:
            text_available += 1
        pattern_counter[
            f"{int(modalities['image_available'])}{int(modalities['text_available'])}"
        ] += 1

        if not record["sample_id"] or not record["label"]:
            missing_fields += 1

        if path_checks:
            candidates = _resolve_image_candidates(sample.get("image"), dataset_dir=dataset_dir)
            if sample.get("image") is not None and not any(candidate.exists() for candidate in candidates):
                invalid_paths.append(record["sample_id"] or f"{client_name}:{idx}")

    total = len(samples)
    dominant_pattern = "unknown"
    if pattern_counter:
        code = pattern_counter.most_common(1)[0][0]
        dominant_pattern = {
            "11": "multimodal",
            "10": "image_only",
            "01": "text_only",
            "00": "no_modality",
        }.get(code, code)

    return {
        "client_name": client_name,
        "num_samples": total,
        "label_distribution": label_counts,
        "image_available": image_available,
        "text_available": text_available,
        "image_missing_ratio": float(1.0 - (image_available / total)) if total else 0.0,
        "text_missing_ratio": float(1.0 - (text_available / total)) if total else 0.0,
        "pattern_histogram": dict(pattern_counter),
        "dominant_pattern": dominant_pattern,
        "missing_required_fields": missing_fields,
        "invalid_image_paths": invalid_paths,
    }, invalid_paths


def summarize_federated_dataset(dataset_dir: Path, path_checks: bool = True) -> Dict[str, Any]:
    client_files = list_client_files(dataset_dir)
    if not client_files:
        raise ValueError(f"No client_*.json files found in {dataset_dir}")

    per_client: List[Dict[str, Any]] = []
    global_label_counter = Counter()
    invalid_path_samples: List[str] = []

    for client_file in client_files:
        samples = load_client_samples(dataset_dir, client_file)
        client_stats, invalids = _client_stats(
            client_file.stem,
            samples,
            dataset_dir=dataset_dir,
            path_checks=path_checks,
        )
        per_client.append(client_stats)
        global_label_counter.update(client_stats["label_distribution"])
        invalid_path_samples.extend(invalids)

    client_sizes = np.asarray([item["num_samples"] for item in per_client], dtype=np.float64)
    pattern_counter = Counter(item["dominant_pattern"] for item in per_client)

    stats = {
        "dataset_dir": str(dataset_dir),
        "dataset_type": "federated",
        "num_clients": len(per_client),
        "total_samples": int(sum(item["num_samples"] for item in per_client)),
        "client_sizes": {item["client_name"]: int(item["num_samples"]) for item in per_client},
        "client_size_summary": {
            "min": int(client_sizes.min()) if client_sizes.size else 0,
            "max": int(client_sizes.max()) if client_sizes.size else 0,
            "mean": float(client_sizes.mean()) if client_sizes.size else 0.0,
            "std": float(client_sizes.std(ddof=0)) if client_sizes.size else 0.0,
        },
        "global_label_distribution": {
            label: int(global_label_counter.get(label, 0)) for label in HUMANITARIAN_LABELS
        },
        "cross_client_pattern_distribution": dict(pattern_counter),
        "per_client": per_client,
    }
    validation = {
        "has_empty_client": any(item["num_samples"] == 0 for item in per_client),
        "empty_clients": [item["client_name"] for item in per_client if item["num_samples"] == 0],
        "invalid_image_path_count": len(invalid_path_samples),
        "invalid_image_path_samples": invalid_path_samples[:200],
        "missing_required_field_clients": [
            item["client_name"] for item in per_client if item["missing_required_fields"] > 0
        ],
        "manifest_exists": (dataset_dir / MANIFEST_FILENAME).exists(),
    }
    return {"stats": stats, "validation": validation}


def summarize_standardized_dataset(dataset_dir: Path, path_checks: bool = True) -> Dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    split_files = manifest.get("split_files", {})
    split_stats: Dict[str, Any] = {}
    invalid_paths: List[str] = []
    for split_name, file_name in split_files.items():
        records = read_json(dataset_dir / file_name)
        label_counts = label_histogram(records)
        split_invalids: List[str] = []
        if path_checks:
            for record in records:
                candidates = _resolve_image_candidates(record.get("image_path"), dataset_dir=dataset_dir)
                if record.get("image_path") and not any(candidate.exists() for candidate in candidates):
                    split_invalids.append(record["sample_id"])
        invalid_paths.extend(split_invalids)
        split_stats[split_name] = {
            "num_samples": len(records),
            "label_distribution": label_counts,
            "invalid_image_paths": split_invalids[:200],
        }
    stats = {
        "dataset_dir": str(dataset_dir),
        "dataset_type": "standardized",
        "split_stats": split_stats,
    }
    validation = {
        "invalid_image_path_count": len(invalid_paths),
        "invalid_image_path_samples": invalid_paths[:200],
        "manifest_exists": True,
    }
    return {"stats": stats, "validation": validation}


def summarize_dataset(dataset_dir: Path, path_checks: bool = True) -> Dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    dataset_type = str(manifest.get("dataset_type", ""))
    if dataset_type.startswith("standardized"):
        return summarize_standardized_dataset(dataset_dir, path_checks=path_checks)
    return summarize_federated_dataset(dataset_dir, path_checks=path_checks)


def write_dataset_reports(dataset_dir: Path, path_checks: bool = False) -> Dict[str, Any]:
    summary = summarize_dataset(dataset_dir, path_checks=path_checks)
    write_json(dataset_dir / STATS_FILENAME, summary["stats"])
    write_json(dataset_dir / VALIDATION_FILENAME, summary["validation"])
    return summary
