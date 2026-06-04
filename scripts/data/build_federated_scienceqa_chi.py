#!/usr/bin/env python3
"""Build ScienceQA federated CHI splits from the image-present parquet subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


LABEL_LEVEL_ALPHA = {
    "L0": 10.0,
    "L1": 0.5,
    "L2": 0.1,
}
MODALITY_LEVEL_RATIOS = {
    "M0": {"full": 1.0, "text_only": 0.0, "image_only": 0.0},
    "M1": {"full": 0.5, "text_only": 0.25, "image_only": 0.25},
    "M2": {"full": 0.2, "text_only": 0.4, "image_only": 0.4},
}
DEFAULT_SETTINGS = ["L0_M0", "L1_M0", "L0_M1", "L1_M1"]
SPLIT_FILE_HINTS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build federated CHI ScienceQA splits from image-present parquet files."
    )
    parser.add_argument("--input_dir", type=Path, default=Path("ScienceQA/image_present/data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/scienceqa/federated_chi"))
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS)
    parser.add_argument(
        "--split_target",
        choices=["answer", "subject", "topic", "category", "skill"],
        default="answer",
    )
    parser.add_argument("--alpha_l1", type=float, default=0.5)
    parser.add_argument("--alpha_l2", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--image_format", choices=["png"], default="png")
    return parser.parse_args()


def stable_id(prefix: str, idx: int, row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "prefix": prefix,
            "idx": idx,
            "question": row.get("question"),
            "answer": row.get("answer"),
            "choices": row.get("choices"),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8", errors="ignore")
    return f"{prefix}_{hashlib.sha1(payload).hexdigest()[:16]}"


def find_split_file(input_dir: Path, split: str) -> Path:
    hint = SPLIT_FILE_HINTS[split]
    matches = sorted(input_dir.glob(f"*{hint}*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet file for split={split!r} under {input_dir}")
    return matches[0]


def read_parquet_records(input_dir: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    path = find_split_file(input_dir, split)
    records = pq.read_table(path).to_pylist()
    if limit is not None and int(limit) > 0:
        records = records[: int(limit)]
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def export_image(row: dict[str, Any], image_dir: Path, sample_id: str, image_format: str) -> str:
    image = row.get("image") or {}
    image_bytes = image.get("bytes") if isinstance(image, dict) else None
    if not image_bytes:
        raise ValueError(f"Sample {sample_id} has no image bytes.")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample_id}.{image_format}"
    if not image_path.exists():
        image_path.write_bytes(image_bytes)
    return str(image_path.as_posix())


def normalize_record(
    row: dict[str, Any],
    split: str,
    idx: int,
    image_dir: Path,
    image_format: str,
) -> dict[str, Any]:
    sample_id = stable_id(split, idx, row)
    choices = normalize_choices(row.get("choices"))
    answer = int(row.get("answer"))
    if answer < 0 or answer >= len(choices):
        raise ValueError(f"Invalid answer index for {sample_id}: answer={answer}, choices={choices}")

    image_path = export_image(row, image_dir=image_dir, sample_id=sample_id, image_format=image_format)
    return {
        "id": sample_id,
        "question": str(row.get("question") or "").strip(),
        "choices": choices,
        "answer": answer,
        "answer_letter": chr(ord("A") + answer),
        "subject": str(row.get("subject") or ""),
        "topic": str(row.get("topic") or ""),
        "category": str(row.get("category") or ""),
        "skill": str(row.get("skill") or ""),
        "task": str(row.get("task") or ""),
        "grade": str(row.get("grade") or ""),
        "image": image_path,
        "hint": str(row.get("hint") or ""),
        "lecture": str(row.get("lecture") or ""),
        "solution": str(row.get("solution") or ""),
        "source_split": split,
    }


def load_normalized_split(
    input_dir: Path,
    split: str,
    output_dir: Path,
    limit: int | None,
    image_format: str,
) -> list[dict[str, Any]]:
    raw_records = read_parquet_records(input_dir=input_dir, split=split, limit=limit)
    image_dir = output_dir / "images" / split
    return [
        normalize_record(
            row=row,
            split=split,
            idx=idx,
            image_dir=image_dir,
            image_format=image_format,
        )
        for idx, row in enumerate(raw_records)
    ]


def label_key(sample: dict[str, Any], split_target: str) -> str:
    value = sample.get(split_target)
    if split_target == "answer":
        value = sample.get("answer")
    text = str(value)
    return text if text else "UNKNOWN"


def build_iid_split(
    samples: list[dict[str, Any]],
    num_clients: int,
    split_target: str,
    seed: int,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[label_key(sample, split_target)].append(sample)
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(num_clients)]
    for offset, key in enumerate(sorted(grouped)):
        items = list(grouped[key])
        rng.shuffle(items)
        for idx, item in enumerate(items):
            clients[(idx + offset) % num_clients].append(dict(item))
    for client in clients:
        rng.shuffle(client)
    return clients


def sample_dirichlet_split(
    samples: list[dict[str, Any]],
    num_clients: int,
    split_target: str,
    alpha: float,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[label_key(sample, split_target)].append(sample)
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(num_clients)]
    for key in sorted(grouped):
        items = list(grouped[key])
        rng.shuffle(items)
        probs = rng.dirichlet(np.full(num_clients, float(alpha), dtype=np.float64))
        counts = rng.multinomial(len(items), probs)
        start = 0
        for cid, count in enumerate(counts.tolist()):
            if count:
                clients[cid].extend(dict(x) for x in items[start:start + count])
            start += count
    repair_empty_clients(clients, rng)
    for client in clients:
        rng.shuffle(client)
    return clients


def repair_empty_clients(clients: list[list[dict[str, Any]]], rng: np.random.Generator) -> None:
    while any(len(client) == 0 for client in clients):
        receiver = min(range(len(clients)), key=lambda cid: len(clients[cid]))
        donor = max(range(len(clients)), key=lambda cid: len(clients[cid]))
        if len(clients[donor]) <= 1:
            return
        move_idx = int(rng.integers(0, len(clients[donor])))
        clients[receiver].append(clients[donor].pop(move_idx))


def parse_setting(setting: str) -> tuple[str, str]:
    parts = setting.strip().upper().split("_")
    if len(parts) != 2 or parts[0] not in LABEL_LEVEL_ALPHA or parts[1] not in MODALITY_LEVEL_RATIOS:
        raise ValueError(f"Invalid setting={setting!r}; expected e.g. L0_M0, L1_M1, L2_M2")
    return parts[0], parts[1]


def client_modality_types(num_clients: int, modality_level: str, seed: int) -> list[str]:
    ratios = MODALITY_LEVEL_RATIOS[modality_level]
    keys = ["full", "text_only", "image_only"]
    if modality_level == "M0":
        return ["full" for _ in range(num_clients)]
    counts = {key: int(math.floor(ratios[key] * num_clients)) for key in keys}
    remaining = num_clients - sum(counts.values())
    ranked = sorted(keys, key=lambda k: ratios[k] - counts[k] / max(1, num_clients), reverse=True)
    for key in ranked[:remaining]:
        counts[key] += 1
    values = []
    for key in keys:
        values.extend([key] * counts[key])
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return values[:num_clients]


def apply_modality(
    clients: list[list[dict[str, Any]]],
    label_level: str,
    modality_level: str,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    modality_types = client_modality_types(len(clients), modality_level, seed=seed)
    out = []
    for cid, samples in enumerate(clients):
        modality_type = modality_types[cid]
        has_image = modality_type in {"full", "image_only"}
        has_text = modality_type in {"full", "text_only"}
        client_samples = []
        for sample in samples:
            item = dict(sample)
            item["client_id"] = int(cid)
            item["label_heterogeneity"] = label_level
            item["modality_heterogeneity"] = modality_level
            item["client_modality_type"] = modality_type
            item["available_context"] = {
                "question": True,
                "image": bool(has_image),
                "text_context": bool(has_text),
            }
            client_samples.append(item)
        out.append(client_samples)
    return out, modality_types


def count_distribution(samples: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(sample.get(key, "UNKNOWN")) for sample in samples)
    return {k: int(v) for k, v in sorted(counter.items())}


def context_key(sample: dict[str, Any]) -> str:
    ctx = sample.get("available_context") or {}
    if ctx.get("image") and ctx.get("text_context"):
        return "full"
    if ctx.get("image"):
        return "image_only"
    if ctx.get("text_context"):
        return "text_only"
    return "question_only"


def build_meta(
    setting: str,
    label_level: str,
    modality_level: str,
    alpha: float,
    split_target: str,
    clients: list[list[dict[str, Any]]],
    modality_types: list[str],
    seed: int,
) -> dict[str, Any]:
    flat = [sample for client in clients for sample in client]
    client_sample_counts = {f"client_{i}": int(len(samples)) for i, samples in enumerate(clients)}
    client_label_distribution = {
        f"client_{i}": count_distribution(samples, split_target if split_target != "answer" else "answer")
        for i, samples in enumerate(clients)
    }
    client_context_distribution = {
        f"client_{i}": dict(Counter(context_key(sample) for sample in samples))
        for i, samples in enumerate(clients)
    }
    return {
        "dataset": "ScienceQA/image_present",
        "setting": setting,
        "num_clients": len(clients),
        "seed": int(seed),
        "label_level": label_level,
        "modality_level": modality_level,
        "alpha": float(alpha),
        "split_target": split_target,
        "modality_ratio": MODALITY_LEVEL_RATIOS[modality_level],
        "client_sample_counts": client_sample_counts,
        "client_label_distribution": client_label_distribution,
        "client_modality_type": {f"client_{i}": modality_types[i] for i in range(len(clients))},
        "client_available_context_distribution": client_context_distribution,
        "global_label_distribution": count_distribution(flat, split_target if split_target != "answer" else "answer"),
        "global_modality_distribution": dict(Counter(context_key(sample) for sample in flat)),
    }


def build_setting(
    train_samples: list[dict[str, Any]],
    output_dir: Path,
    setting: str,
    num_clients: int,
    split_target: str,
    alpha_l1: float,
    alpha_l2: float,
    seed: int,
) -> None:
    label_level, modality_level = parse_setting(setting)
    if label_level == "L1":
        alpha = float(alpha_l1)
    elif label_level == "L2":
        alpha = float(alpha_l2)
    else:
        alpha = LABEL_LEVEL_ALPHA[label_level]

    setting_seed = int(hashlib.sha1(f"{seed}:{setting}".encode("utf-8")).hexdigest()[:8], 16)
    if label_level == "L0":
        clients = build_iid_split(train_samples, num_clients, split_target, seed=setting_seed)
    else:
        clients = sample_dirichlet_split(train_samples, num_clients, split_target, alpha, seed=setting_seed)
    clients, modality_types = apply_modality(clients, label_level, modality_level, seed=setting_seed + 17)

    setting_dir = output_dir / setting
    for cid, samples in enumerate(clients):
        write_jsonl(setting_dir / f"client_{cid}.jsonl", samples)
    meta = build_meta(
        setting=setting,
        label_level=label_level,
        modality_level=modality_level,
        alpha=alpha,
        split_target=split_target,
        clients=clients,
        modality_types=modality_types,
        seed=seed,
    )
    write_json(setting_dir / "meta.json", meta)


def annotate_eval(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for sample in samples:
        item = dict(sample)
        item["client_id"] = None
        item["label_heterogeneity"] = "eval"
        item["modality_heterogeneity"] = "eval"
        item["client_modality_type"] = "full"
        item["available_context"] = {"question": True, "image": True, "text_context": True}
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    if args.num_clients <= 0:
        raise ValueError("--num_clients must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_normalized_split(
        input_dir=args.input_dir,
        split="train",
        output_dir=args.output_dir,
        limit=args.max_train_samples,
        image_format=args.image_format,
    )
    validation = load_normalized_split(
        input_dir=args.input_dir,
        split="validation",
        output_dir=args.output_dir,
        limit=args.max_eval_samples,
        image_format=args.image_format,
    )
    test = load_normalized_split(
        input_dir=args.input_dir,
        split="test",
        output_dir=args.output_dir,
        limit=args.max_eval_samples,
        image_format=args.image_format,
    )

    write_jsonl(args.output_dir / "validation.jsonl", annotate_eval(validation))
    write_jsonl(args.output_dir / "test.jsonl", annotate_eval(test))

    for setting in args.settings:
        build_setting(
            train_samples=train,
            output_dir=args.output_dir,
            setting=setting,
            num_clients=args.num_clients,
            split_target=args.split_target,
            alpha_l1=args.alpha_l1,
            alpha_l2=args.alpha_l2,
            seed=args.seed,
        )

    write_json(
        args.output_dir / "build_summary.json",
        {
            "dataset": "ScienceQA/image_present",
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "settings": args.settings,
            "num_clients": args.num_clients,
            "split_target": args.split_target,
            "seed": args.seed,
            "num_train_samples": len(train),
            "num_validation_samples": len(validation),
            "num_test_samples": len(test),
        },
    )
    print(f"Wrote ScienceQA CHI splits -> {args.output_dir}")


if __name__ == "__main__":
    main()
