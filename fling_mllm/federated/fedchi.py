"""FedCHI adapter grouping, activation, and heterogeneity metadata utilities."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterable

import torch


ADAPTER_TYPES = ("shared", "image", "text")
SCIENCEQA_SEMANTIC_KEYS = ("subject", "topic", "category", "skill")


def infer_adapter_type(param_name: str) -> str:
    lowered = param_name.lower()
    if any(token in lowered for token in (".image.", "image_adapter", "lora_image")):
        return "image"
    if any(token in lowered for token in (".text.", "text_adapter", "lora_text")):
        return "text"
    if any(token in lowered for token in (".shared.", "shared_adapter", "lora_shared")):
        return "shared"
    return "shared"


def split_state_by_adapter(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    grouped = {adapter_type: {} for adapter_type in ADAPTER_TYPES}
    for key, value in state_dict.items():
        grouped[infer_adapter_type(key)][key] = value
    return grouped


def adapter_names_for_modality(client_modality_type: str, enable_shared=True, enable_image=True, enable_text=True):
    modality = str(client_modality_type or "full").lower()
    names = []
    if enable_shared:
        names.append("shared")
    if enable_image and modality in {"full", "image_only", "image"}:
        names.append("image")
    if enable_text and modality in {"full", "text_only", "text"}:
        names.append("text")
    return names or ["shared"]


def set_fedchi_active_adapters(model, adapter_names: list[str]) -> None:
    if not hasattr(model, "set_adapter"):
        return
    try:
        model.set_adapter(adapter_names)
        return
    except Exception:
        pass
    # Compatibility fallback for PEFT versions that accept one active adapter.
    try:
        model.set_adapter(adapter_names[0])
    except Exception:
        return


def modality_from_samples(samples: Iterable[dict]) -> str:
    counter = Counter(str(sample.get("client_modality_type", "full")) for sample in samples)
    if not counter:
        return "full"
    return counter.most_common(1)[0][0]


def _distribution(samples: Iterable[dict], key: str, universe: list[str]) -> dict[str, float]:
    counter = Counter(str(sample.get(key, "UNKNOWN")) for sample in samples)
    total = float(sum(counter.values()))
    if total <= 0:
        return {item: 0.0 for item in universe}
    return {item: float(counter.get(item, 0)) / total for item in universe}


def _infer_semantic_key(local_samples: list[list[dict]]) -> str:
    flat = [sample for samples in local_samples for sample in samples]
    for sample in flat:
        target = sample.get("semantic_split_target")
        if target:
            return str(target)
    env_target = os.environ.get("SCIENCEQA_SPLIT_TARGET")
    if env_target:
        return str(env_target)
    for key in SCIENCEQA_SEMANTIC_KEYS:
        if any(sample.get(key) not in {None, ""} for sample in flat):
            return key
    return "answer"


def _l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    return float(sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys))


def _context_key(sample: dict) -> str:
    ctx = sample.get("available_context") or {}
    if ctx.get("image") and ctx.get("text_context"):
        return "full"
    if ctx.get("image"):
        return "image_only"
    if ctx.get("text_context"):
        return "text_only"
    return "question_only"


def build_fedchi_info(
    local_samples: list[list[dict]],
    lambda_label: float = 1.0,
    lambda_modality: float = 1.0,
    semantic_key: str | None = None,
) -> dict:
    semantic_key = str(semantic_key or _infer_semantic_key(local_samples))
    semantic_values = sorted({
        str(sample.get(semantic_key, "UNKNOWN"))
        for samples in local_samples
        for sample in samples
    })
    modalities = ["full", "image_only", "text_only", "question_only"]
    flat = [sample for samples in local_samples for sample in samples]
    global_semantic = _distribution(flat, semantic_key, semantic_values)
    global_modality_counter = Counter(_context_key(sample) for sample in flat)
    modality_total = float(sum(global_modality_counter.values()))
    global_modality = {
        key: (float(global_modality_counter.get(key, 0)) / modality_total if modality_total else 0.0)
        for key in modalities
    }

    client_weight_divisor = {}
    client_heterogeneity = {}
    client_modality_type = {}
    client_has_image = {}
    client_has_text = {}
    for client_idx, samples in enumerate(local_samples):
        semantic_div = _l1(_distribution(samples, semantic_key, semantic_values), global_semantic)
        modality_counter = Counter(_context_key(sample) for sample in samples)
        modality_total = float(sum(modality_counter.values()))
        modality_dist = {
            key: (float(modality_counter.get(key, 0)) / modality_total if modality_total else 0.0)
            for key in modalities
        }
        modality_div = _l1(modality_dist, global_modality)
        div_k = 1.0 + float(lambda_label) * semantic_div + float(lambda_modality) * modality_div
        modality_type = modality_from_samples(samples)
        client_weight_divisor[int(client_idx)] = float(div_k)
        client_heterogeneity[f"client_{client_idx}"] = {
            "label_div": float(semantic_div),
            "semantic_div": float(semantic_div),
            "semantic_key": semantic_key,
            "modality_div": float(modality_div),
            "div_k": float(div_k),
        }
        client_modality_type[int(client_idx)] = modality_type
        client_has_image[int(client_idx)] = modality_type in {"full", "image_only", "image"}
        client_has_text[int(client_idx)] = modality_type in {"full", "text_only", "text"}

    return {
        "client_weight_divisor": client_weight_divisor,
        "client_heterogeneity": client_heterogeneity,
        "client_modality_type": client_modality_type,
        "client_has_image": client_has_image,
        "client_has_text": client_has_text,
        "semantic_key": semantic_key,
        "global_label_distribution": global_semantic,
        "global_semantic_distribution": global_semantic,
        "global_modality_distribution": global_modality,
        "lambda_label": float(lambda_label),
        "lambda_modality": float(lambda_modality),
    }


def append_fedchi_weight_records(output_path: str | None, records: list[dict]) -> None:
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
