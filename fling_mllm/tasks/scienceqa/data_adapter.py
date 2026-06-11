"""ScienceQA data loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..registry import register_data_loader
from .prompt_builder import answer_letter_from_sample, build_scienceqa_prompt, image_path_for_context


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "samples", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"ScienceQA JSON must be a list or contain records/samples/data: {path}")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    out = []
    for idx, row in enumerate(rows):
        image = row.get("image")
        out.append({
            "id": f"{path.stem}_{idx}",
            "question": row.get("question"),
            "choices": row.get("choices"),
            "answer": row.get("answer"),
            "subject": row.get("subject"),
            "topic": row.get("topic"),
            "category": row.get("category"),
            "skill": row.get("skill"),
            "hint": row.get("hint"),
            "lecture": row.get("lecture"),
            "solution": row.get("solution"),
            "image": image.get("path") if isinstance(image, dict) else None,
            "available_context": {"question": True, "image": True, "text_context": True},
            "client_modality_type": "full",
        })
    return out


def _load_records(data_path: str, data_format: str) -> list[dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"ScienceQA data path not found: {data_path}")
    fmt = str(data_format or "auto").lower()
    if fmt == "auto":
        fmt = path.suffix.lower().lstrip(".")
    if fmt == "jsonl":
        return _read_jsonl(path)
    if fmt == "json":
        return _read_json(path)
    if fmt == "parquet":
        return _read_parquet(path)
    raise ValueError(f"Unsupported ScienceQA data_format={data_format!r} for path={data_path}")


def normalize_scienceqa_sample(sample: dict[str, Any], require_answer: bool = True) -> dict[str, Any]:
    item = dict(sample)
    if "available_context" not in item or not isinstance(item.get("available_context"), dict):
        item["available_context"] = {"question": True, "image": bool(item.get("image")), "text_context": True}
    item["prompt"] = build_scienceqa_prompt(item)
    item["question"] = item["prompt"]
    item["image"] = image_path_for_context(item)
    if require_answer:
        item["answer_letter"] = answer_letter_from_sample(item)
        item["answer"] = item["answer_letter"]
    elif "answer" in item and item.get("answer") is not None:
        item["answer_letter"] = answer_letter_from_sample(item)
    return item


@register_data_loader("scienceqa")
def load_scienceqa_samples(
    data_path: str,
    split: str = "train",
    data_format: str = "auto",
    require_answer: bool = True,
    max_samples: int | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    del split, kwargs
    records = _load_records(data_path, data_format=data_format)
    if max_samples is not None and int(max_samples) > 0:
        records = records[: int(max_samples)]
    return [normalize_scienceqa_sample(sample, require_answer=require_answer) for sample in records]
