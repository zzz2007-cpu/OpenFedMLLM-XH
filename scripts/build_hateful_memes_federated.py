#!/usr/bin/env python3
"""Build federated Hateful Memes JSONL partitions.

The script reads only train.jsonl, keeps image paths relative, and writes
client-level JSONL files plus metadata for statistical and modal settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IMAGE_FIELDS = ("img", "image")
TEXT_FIELDS = ("text", "sentence")
LABEL_FIELDS = ("label", "labels")
ID_FIELDS = ("id", "sample_id")
MAX_DIRICHLET_RETRIES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build federated partitions for Hateful Memes.")
    parser.add_argument("--data_dir", type=Path, default=Path("hateful_memes"))
    parser.add_argument("--out_dir", type=Path, default=Path("hateful_memes/federated"))
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.1])
    parser.add_argument("--missing_rates", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    parser.add_argument("--cross_ratios", type=str, nargs="+", default=["3:7", "5:5", "7:3"])
    parser.add_argument("--hybrid_keep_probs", type=float, nargs="+", default=[0.8, 0.7, 0.6])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_field(columns: set[str], candidates: tuple[str, ...], name: str, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(
            f"Missing required {name} field. Tried {list(candidates)}; available fields: {sorted(columns)}"
        )
    return None


def normalize_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_records:
        raise ValueError("No records found in train.jsonl.")

    columns = set().union(*(record.keys() for record in raw_records))
    id_field = find_field(columns, ID_FIELDS, "id", required=False)
    img_field = find_field(columns, IMAGE_FIELDS, "image")
    text_field = find_field(columns, TEXT_FIELDS, "text")
    label_field = find_field(columns, LABEL_FIELDS, "label")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, record in enumerate(raw_records):
        sample_id = record.get(id_field) if id_field else None
        sample_id = str(sample_id) if sample_id is not None and str(sample_id).strip() else f"train_{idx:06d}"
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id found: {sample_id}")
        seen_ids.add(sample_id)

        img = record.get(img_field)
        text = record.get(text_field)
        label = record.get(label_field)
        if img is None or str(img).strip() == "":
            raise ValueError(f"Record {sample_id} has empty image path.")
        if text is None:
            text = ""
        if label is None:
            raise ValueError(f"Record {sample_id} has missing label.")

        records.append(
            {
                "id": sample_id,
                "img": str(img),
                "text": str(text),
                "label": int(label),
            }
        )
    return records


def format_float(value: float) -> str:
    return str(float(value))


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "::".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def copy_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "img": sample["img"],
        "text": sample["text"],
        "label": int(sample["label"]),
    }


def group_by_label(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["label"])].append(record)
    return dict(grouped)


def shuffle_clients(clients: list[list[dict[str, Any]]], rng: np.random.Generator) -> None:
    for client_records in clients:
        rng.shuffle(client_records)


def build_iid_split(
    records: list[dict[str, Any]],
    num_clients: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if num_clients > len(records):
        raise ValueError("--num_clients cannot exceed the number of training samples.")

    rng = np.random.default_rng(seed)
    clients: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    grouped = group_by_label(records)

    for label_idx, label in enumerate(sorted(grouped)):
        label_records = list(grouped[label])
        rng.shuffle(label_records)
        offset = label_idx % num_clients
        for idx, record in enumerate(label_records):
            clients[(idx + offset) % num_clients].append(record)

    shuffle_clients(clients, rng)
    return clients


def candidate_score(clients: list[list[dict[str, Any]]]) -> tuple[int, int]:
    sizes = [len(client) for client in clients]
    return min(sizes), -int(np.std(sizes))


def sample_dirichlet_once(
    grouped: dict[int, list[dict[str, Any]]],
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[dict[str, Any]]]:
    clients: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for label in sorted(grouped):
        label_records = list(grouped[label])
        rng.shuffle(label_records)
        probs = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
        counts = rng.multinomial(len(label_records), probs)
        start = 0
        for cid, count in enumerate(counts.tolist()):
            end = start + count
            if count:
                clients[cid].extend(label_records[start:end])
            start = end
    return clients


def repair_small_clients(
    clients: list[list[dict[str, Any]]],
    min_samples: int,
    rng: np.random.Generator,
) -> None:
    while True:
        sizes = [len(client) for client in clients]
        small_clients = [cid for cid, size in enumerate(sizes) if size < min_samples]
        if not small_clients:
            return

        receiver = min(small_clients, key=lambda cid: sizes[cid])
        donor = max(range(len(clients)), key=lambda cid: sizes[cid])
        if sizes[donor] <= min_samples:
            return

        move_idx = int(rng.integers(0, len(clients[donor])))
        clients[receiver].append(clients[donor].pop(move_idx))


def build_dirichlet_split(
    records: list[dict[str, Any]],
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if num_clients <= 0:
        raise ValueError("--num_clients must be positive.")
    if num_clients > len(records):
        raise ValueError("--num_clients cannot exceed the number of training samples.")

    rng = np.random.default_rng(seed)
    grouped = group_by_label(records)
    min_samples = max(1, int(math.floor((len(records) / num_clients) * 0.05)))
    best_clients: list[list[dict[str, Any]]] | None = None
    best_score: tuple[int, int] | None = None

    for _ in range(MAX_DIRICHLET_RETRIES):
        clients = sample_dirichlet_once(grouped, num_clients, alpha, rng)
        score = candidate_score(clients)
        if best_score is None or score > best_score:
            best_clients = clients
            best_score = score
        if all(len(client) >= min_samples for client in clients):
            shuffle_clients(clients, rng)
            return clients

    if best_clients is None:
        raise RuntimeError("Failed to sample a Dirichlet partition.")
    repair_small_clients(best_clients, min_samples, rng)
    shuffle_clients(best_clients, rng)
    return best_clients


def clone_clients(clients: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return [[copy_sample(sample) for sample in client] for client in clients]


def apply_aligned(
    clients: list[list[dict[str, Any]]],
    rng: np.random.Generator | None = None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    return clone_clients(clients), {}


def apply_missing(
    clients: list[list[dict[str, Any]]],
    missing_rate: float,
    rng: np.random.Generator,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if not 0 <= missing_rate <= 1:
        raise ValueError("Missing rate must be in [0, 1].")
    output = clone_clients(clients)
    for client_records in output:
        for sample in client_records:
            keep_img = bool(rng.random() >= missing_rate)
            keep_text = bool(rng.random() >= missing_rate)
            if not keep_img and not keep_text:
                if bool(rng.integers(0, 2)):
                    keep_img = True
                else:
                    keep_text = True
            if not keep_img:
                sample["img"] = None
            if not keep_text:
                sample["text"] = None
    return output, {"missing_rate": missing_rate}


def parse_cross_ratio(ratio: str, num_clients: int) -> tuple[int, int]:
    parts = ratio.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid cross ratio {ratio!r}; expected format like 3:7.")
    image_only = int(parts[0])
    text_only = int(parts[1])
    if image_only < 0 or text_only < 0:
        raise ValueError(f"Invalid cross ratio {ratio!r}; counts must be non-negative.")
    if image_only + text_only != num_clients:
        raise ValueError(
            f"Cross ratio {ratio!r} sums to {image_only + text_only}, but num_clients is {num_clients}."
        )
    return image_only, text_only


def apply_cross(
    clients: list[list[dict[str, Any]]],
    image_only_count: int,
    text_only_count: int,
    rng: np.random.Generator,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    num_clients = len(clients)
    if image_only_count + text_only_count != num_clients:
        raise ValueError("image_only_count + text_only_count must equal number of clients.")

    order = rng.permutation(num_clients).tolist()
    image_only_clients = sorted(order[:image_only_count])
    text_only_clients = sorted(order[image_only_count:])
    image_only_set = set(image_only_clients)

    output = clone_clients(clients)
    for cid, client_records in enumerate(output):
        for sample in client_records:
            if cid in image_only_set:
                sample["text"] = None
            else:
                sample["img"] = None

    return output, {
        "image_only_clients": [f"client_{cid}" for cid in image_only_clients],
        "text_only_clients": [f"client_{cid}" for cid in text_only_clients],
    }


def apply_hybrid(
    clients: list[list[dict[str, Any]]],
    keep_prob: float,
    rng: np.random.Generator,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if not 0 <= keep_prob <= 1:
        raise ValueError("Hybrid keep probability must be in [0, 1].")
    output = clone_clients(clients)
    for client_records in output:
        for sample in client_records:
            keep_img = bool(rng.random() < keep_prob)
            keep_text = bool(rng.random() < keep_prob)
            if not keep_img and not keep_text:
                if bool(rng.integers(0, 2)):
                    keep_img = True
                else:
                    keep_text = True
            if not keep_img:
                sample["img"] = None
            if not keep_text:
                sample["text"] = None
    return output, {"hybrid_keep_prob": keep_prob}


def modality_key(sample: dict[str, Any]) -> str:
    has_img = sample.get("img") is not None
    has_text = sample.get("text") is not None
    if has_img and has_text:
        return "image_text"
    if has_img:
        return "image_only"
    if has_text:
        return "text_only"
    return "none"


def compute_meta(
    clients: list[list[dict[str, Any]]],
    dataset: str,
    source_split: str,
    num_clients: int,
    seed: int,
    stat_setting: str,
    modal_setting: str,
    alpha: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = sorted({int(sample["label"]) for client in clients for sample in client})
    num_samples_per_client = {f"client_{cid}": len(client) for cid, client in enumerate(clients)}
    label_distribution_per_client: dict[str, dict[str, int]] = {}
    modality_distribution_per_client: dict[str, dict[str, int]] = {}

    for cid, client in enumerate(clients):
        label_counter = Counter(int(sample["label"]) for sample in client)
        modality_counter = Counter(modality_key(sample) for sample in client)
        label_distribution_per_client[f"client_{cid}"] = {
            str(label): int(label_counter.get(label, 0)) for label in labels
        }
        modality_distribution_per_client[f"client_{cid}"] = {
            "image_text": int(modality_counter.get("image_text", 0)),
            "image_only": int(modality_counter.get("image_only", 0)),
            "text_only": int(modality_counter.get("text_only", 0)),
        }

    global_label_counter = Counter(int(sample["label"]) for client in clients for sample in client)
    meta = {
        "dataset": dataset,
        "source_split": source_split,
        "num_clients": num_clients,
        "seed": seed,
        "stat_setting": stat_setting,
        "modal_setting": modal_setting,
        "alpha": alpha,
        "num_samples_total": int(sum(num_samples_per_client.values())),
        "num_samples_per_client": num_samples_per_client,
        "label_distribution_per_client": label_distribution_per_client,
        "modality_distribution_per_client": modality_distribution_per_client,
        "global_label_distribution": {str(label): int(global_label_counter.get(label, 0)) for label in labels},
        "empty_clients": [client for client, count in num_samples_per_client.items() if count == 0],
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")


def clean_setting_dir(setting_dir: Path) -> None:
    setting_dir.mkdir(parents=True, exist_ok=True)
    for old_file in setting_dir.glob("client_*.jsonl"):
        old_file.unlink()
    meta_path = setting_dir / "meta.json"
    if meta_path.exists():
        meta_path.unlink()


def print_setting_log(meta: dict[str, Any]) -> None:
    print(
        f"[{meta['stat_setting']}/{meta['modal_setting']}] "
        f"samples={meta['num_samples_total']} empty_clients={meta['empty_clients']}"
    )
    print("  label_distribution_per_client:")
    for client, dist in meta["label_distribution_per_client"].items():
        print(f"    {client}: {dist}")
    print("  modality_distribution_per_client:")
    for client, dist in meta["modality_distribution_per_client"].items():
        print(f"    {client}: {dist}")


def write_setting(
    setting_dir: Path,
    clients: list[list[dict[str, Any]]],
    meta: dict[str, Any],
) -> None:
    clean_setting_dir(setting_dir)
    for cid, client_records in enumerate(clients):
        write_jsonl(setting_dir / f"client_{cid}.jsonl", client_records)
    write_meta(setting_dir / "meta.json", meta)
    print_setting_log(meta)


def build_stat_splits(
    records: list[dict[str, Any]],
    num_clients: int,
    seed: int,
    alphas: list[float],
) -> list[tuple[str, float | None, list[list[dict[str, Any]]]]]:
    splits: list[tuple[str, float | None, list[list[dict[str, Any]]]]] = []
    splits.append(("iid", None, build_iid_split(records, num_clients, seed + 0)))
    for idx, alpha in enumerate(alphas, start=1):
        name = f"dir_{format_float(alpha)}"
        clients = build_dirichlet_split(records, num_clients, alpha, seed + idx)
        splits.append((name, float(alpha), clients))
    return splits


def summary_entry(meta: dict[str, Any], relative_dir: Path) -> dict[str, Any]:
    return {
        "path": relative_dir.as_posix(),
        "stat_setting": meta["stat_setting"],
        "modal_setting": meta["modal_setting"],
        "alpha": meta["alpha"],
        "num_samples_total": meta["num_samples_total"],
        "num_samples_per_client": meta["num_samples_per_client"],
        "empty_clients": meta["empty_clients"],
        "label_distribution_per_client": meta["label_distribution_per_client"],
        "modality_distribution_per_client": meta["modality_distribution_per_client"],
        "image_only_clients": meta.get("image_only_clients", []),
        "text_only_clients": meta.get("text_only_clients", []),
    }


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    raw_records = read_jsonl(train_path)
    records = normalize_records(raw_records)
    labels = Counter(record["label"] for record in records)
    print(f"Loaded {len(records)} train samples from {train_path}")
    print(f"Global label distribution: {dict(sorted(labels.items()))}")

    stat_splits = build_stat_splits(records, args.num_clients, args.seed, args.alphas)
    summary: dict[str, Any] = {
        "dataset": "hateful_memes",
        "source_split": "train",
        "num_clients": args.num_clients,
        "seed": args.seed,
        "settings": [],
    }

    for stat_idx, (stat_name, alpha, base_clients) in enumerate(stat_splits):
        print(f"[{stat_name}] generated {sum(len(client) for client in base_clients)} samples")
        modal_jobs: list[tuple[str, list[list[dict[str, Any]]], dict[str, Any]]] = []

        aligned_rng = np.random.default_rng(stable_seed(args.seed, stat_name, "aligned"))
        aligned_clients, aligned_extra = apply_aligned(base_clients, aligned_rng)
        modal_jobs.append(("aligned", aligned_clients, aligned_extra))

        for rate in args.missing_rates:
            modal_name = f"missing_{format_float(rate)}"
            rng = np.random.default_rng(stable_seed(args.seed, stat_name, modal_name))
            clients, extra = apply_missing(base_clients, float(rate), rng)
            modal_jobs.append((modal_name, clients, extra))

        for ratio in args.cross_ratios:
            image_only_count, text_only_count = parse_cross_ratio(ratio, args.num_clients)
            modal_name = f"cross_{image_only_count}_{text_only_count}"
            rng = np.random.default_rng(stable_seed(args.seed, stat_name, modal_name))
            clients, extra = apply_cross(base_clients, image_only_count, text_only_count, rng)
            modal_jobs.append((modal_name, clients, extra))

        for keep_prob in args.hybrid_keep_probs:
            modal_name = f"hybrid_{format_float(keep_prob)}"
            rng = np.random.default_rng(stable_seed(args.seed, stat_name, modal_name))
            clients, extra = apply_hybrid(base_clients, float(keep_prob), rng)
            modal_jobs.append((modal_name, clients, extra))

        for modal_name, modal_clients, extra in modal_jobs:
            meta = compute_meta(
                clients=modal_clients,
                dataset="hateful_memes",
                source_split="train",
                num_clients=args.num_clients,
                seed=args.seed,
                stat_setting=stat_name,
                modal_setting=modal_name,
                alpha=alpha,
                extra=extra,
            )
            setting_dir = args.out_dir / stat_name / modal_name
            write_setting(setting_dir, modal_clients, meta)
            summary["settings"].append(summary_entry(meta, setting_dir.relative_to(args.out_dir)))

        if stat_idx < len(stat_splits) - 1:
            print()

    write_meta(args.out_dir / "summary.json", summary)
    print(f"\nWrote summary to {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
