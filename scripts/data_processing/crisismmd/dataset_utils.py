from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_LABEL_SOURCE,
    HUMANITARIAN_LABELS,
    LETTER_TO_LABEL,
    PROMPT_QUESTIONS,
    QUESTION_BIMODAL,
    QUESTION_IMAGE_ONLY,
    QUESTION_NO_MODALITY,
    QUESTION_TEXT_ONLY,
    STANDARD_SPLIT_FILES,
)
from .io_utils import join_posix, read_json
from .prompting import build_conversations, modality_pattern_name
from .text_cleaning import clean_tweet_text


def normalize_label(text: str) -> str:
    label = (text or "").strip().lower().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^a-z_]", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def parse_assistant_label(answer_text: str) -> str:
    raw = (answer_text or "").strip()
    if not raw:
        raise ValueError("Empty assistant answer.")

    match = re.search(r"\(\s*([A-Ha-h])\s*\)\s*(.+)", raw)
    if match:
        tail = normalize_label(match.group(2))
        if tail in HUMANITARIAN_LABELS:
            return tail
        return LETTER_TO_LABEL[match.group(1).upper()]

    match = re.fullmatch(r"\(?\s*([A-Ha-h])\s*\)?", raw)
    if match:
        return LETTER_TO_LABEL[match.group(1).upper()]

    compact = normalize_label(raw)
    if compact in HUMANITARIAN_LABELS:
        return compact
    raise ValueError(f"Unable to parse humanitarian label from answer: {answer_text!r}")


def extract_prompt_text(user_content: str) -> str:
    lines = [(line or "").strip() for line in (user_content or "").splitlines()]
    if not lines:
        return ""
    if lines[0] == "<image>":
        lines = lines[1:]
    if len(lines) < 2:
        return ""

    question_idx = -1
    for idx, line in enumerate(lines):
        if line in PROMPT_QUESTIONS:
            question_idx = idx
            break

    if question_idx == -1 or question_idx <= 1:
        return ""

    text_lines = [line for line in lines[1:question_idx] if line]
    return "\n".join(text_lines).strip()


def infer_modalities_from_prompt(sample: Dict[str, Any]) -> Tuple[bool, bool]:
    metadata_modalities = (
        sample.get("metadata", {})
        .get("modalities", {})
    )
    if "image_available" in metadata_modalities and "text_available" in metadata_modalities:
        return bool(metadata_modalities["image_available"]), bool(metadata_modalities["text_available"])

    image_available = isinstance(sample.get("image"), str) and bool(sample.get("image"))

    user_content = ""
    for turn in sample.get("conversations", []) or []:
        if isinstance(turn, dict) and str(turn.get("role", "")).lower() == "user":
            user_content = str(turn.get("content", ""))
            break

    if QUESTION_BIMODAL in user_content or QUESTION_TEXT_ONLY in user_content:
        text_available = True
    elif QUESTION_IMAGE_ONLY in user_content or QUESTION_NO_MODALITY in user_content:
        text_available = False
    else:
        text_available = bool(extract_prompt_text(user_content))
    return image_available, text_available


def load_standardized_split(dataset_dir: Path, split: str) -> List[Dict[str, Any]]:
    split_name = STANDARD_SPLIT_FILES[split]
    payload = read_json(dataset_dir / split_name)
    if not isinstance(payload, list):
        raise ValueError(f"Standardized split must be a JSON list: {dataset_dir / split_name}")
    return payload


def load_client_samples(dataset_dir: Path, client_file: Path) -> List[Dict[str, Any]]:
    payload = read_json(client_file)
    if not isinstance(payload, list):
        raise ValueError(f"Client shard must be a JSON list: {client_file}")
    return payload


def standardized_record_from_row(
    row: Dict[str, str],
    split: str,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    label_source: str = DEFAULT_LABEL_SOURCE,
) -> Dict[str, Any]:
    label_value = normalize_label(row.get(label_source, ""))
    if label_value not in HUMANITARIAN_LABELS:
        raise ValueError(
            f"Unsupported humanitarian label from column {label_source!r}: {row.get(label_source)!r}"
        )

    image_rel_path = str(row.get("image", "")).strip().replace("\\", "/")
    image_path = join_posix(image_prefix, image_rel_path) if image_rel_path else None
    cleaned_text = clean_tweet_text(row.get("tweet_text", ""))

    return {
        "sample_id": str(row.get("image_id", "")).strip(),
        "tweet_id": str(row.get("tweet_id", "")).strip(),
        "event_name": str(row.get("event_name", "")).strip(),
        "split": split,
        "text": cleaned_text,
        "raw_text": str(row.get("tweet_text", "")).strip(),
        "image_rel_path": image_rel_path,
        "image_path": image_path,
        "label": label_value,
        "label_source": label_source,
        "label_candidates": {
            "label": normalize_label(row.get("label", "")),
            "label_text": normalize_label(row.get("label_text", "")),
            "label_image": normalize_label(row.get("label_image", "")),
            "label_text_image": str(row.get("label_text_image", "")).strip(),
        },
        "modalities": {
            "image_available": bool(image_rel_path),
            "text_available": bool(cleaned_text),
        },
    }


def read_standardized_tsv(tsv_path: Path, split: str, image_prefix: str, label_source: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            record = standardized_record_from_row(
                row=row,
                split=split,
                image_prefix=image_prefix,
                label_source=label_source,
            )
            if not record["sample_id"]:
                continue
            records.append(record)
    return records


def standardized_to_training_sample(
    record: Dict[str, Any],
    has_image: bool,
    has_text: bool,
    transformation_notes: Optional[Sequence[str]] = None,
    client_id: Optional[int] = None,
) -> Dict[str, Any]:
    mode = modality_pattern_name(has_image=has_image, has_text=has_text)
    notes = list(transformation_notes or [])
    sample = {
        "id": record["sample_id"],
        "image": record["image_path"] if has_image else None,
        "text": record.get("text", ""),
        "conversations": build_conversations(
            text=record.get("text", "") if has_text else "",
            label=record["label"],
            has_image=has_image,
            has_text=has_text,
        ),
        "metadata": {
            "task": "humanitarian",
            "sample_id": record["sample_id"],
            "tweet_id": record.get("tweet_id"),
            "event_name": record.get("event_name"),
            "split": record.get("split", "train"),
            "label": record["label"],
            "label_source": record.get("label_source", DEFAULT_LABEL_SOURCE),
            "client_id": client_id,
            "modalities": {
                "image_available": has_image,
                "text_available": has_text,
                "pattern": mode,
            },
            "source": {
                "text": record.get("text", ""),
                "raw_text": record.get("raw_text", record.get("text", "")),
                "image_rel_path": record.get("image_rel_path"),
                "image_path": record.get("image_path"),
                "label_candidates": record.get("label_candidates", {}),
            },
            "transformations": notes,
        },
    }
    return sample


def sample_to_base_record(sample: Dict[str, Any], default_split: str = "train") -> Dict[str, Any]:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
    source = metadata.get("source", {}) if isinstance(metadata.get("source"), dict) else {}

    user_content = ""
    assistant_content = ""
    for turn in sample.get("conversations", []) or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).lower()
        if role == "user" and not user_content:
            user_content = str(turn.get("content", ""))
        elif role == "assistant" and not assistant_content:
            assistant_content = str(turn.get("content", ""))

    text = source.get("text") or sample.get("text") or extract_prompt_text(user_content)
    label = metadata.get("label")
    if not label:
        label = parse_assistant_label(assistant_content)

    has_image, has_text = infer_modalities_from_prompt(sample)

    return {
        "sample_id": str(sample.get("id") or metadata.get("sample_id") or ""),
        "tweet_id": metadata.get("tweet_id"),
        "event_name": metadata.get("event_name"),
        "split": metadata.get("split", default_split),
        "text": text,
        "raw_text": source.get("raw_text", text),
        "image_rel_path": source.get("image_rel_path"),
        "image_path": source.get("image_path", sample.get("image")),
        "label": normalize_label(label),
        "label_source": metadata.get("label_source", DEFAULT_LABEL_SOURCE),
        "label_candidates": source.get("label_candidates", {}),
        "modalities": {
            "image_available": has_image,
            "text_available": has_text,
        },
        "transformations": list(metadata.get("transformations", [])),
        "client_id": metadata.get("client_id"),
    }


def label_histogram(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for record in records:
        label = record.get("label")
        if label:
            counter[label] += 1
    return {label: int(counter.get(label, 0)) for label in HUMANITARIAN_LABELS}
