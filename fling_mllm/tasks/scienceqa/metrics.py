"""ScienceQA option parsing and metrics."""

from __future__ import annotations

import re
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


_ANSWER_PATTERNS = [
    re.compile(r"(?:answer|option|choice)\s*(?:is|:|：)?\s*\(?\s*([A-Z])\s*\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*([A-Z])\s*\)?(?:[\.\):,\s]|$)", re.IGNORECASE),
]


def parse_scienceqa_prediction(raw_output: Any, num_choices: int) -> str | None:
    text = "" if raw_output is None else str(raw_output).strip()
    if not text:
        return None
    valid = {chr(ord("A") + idx) for idx in range(int(num_choices))}

    stripped = text.strip()
    if stripped.isdigit():
        idx = int(stripped)
        if 0 <= idx < num_choices:
            return chr(ord("A") + idx)

    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter

    upper = text.upper()
    for letter in sorted(valid):
        if re.search(rf"\b{re.escape(letter)}\b", upper):
            return letter
    return None


def compute_scienceqa_metrics(predictions: list[str | None], labels: list[str]) -> dict:
    normalized_preds = [pred if pred is not None else "INVALID" for pred in predictions]
    label_names = sorted(set(labels))
    acc = accuracy_score(labels, normalized_preds) if labels else 0.0
    macro_f1 = f1_score(labels, normalized_preds, labels=label_names, average="macro", zero_division=0) if labels else 0.0
    per_class_f1 = {}
    for label in label_names:
        per_class_f1[label] = f1_score(
            labels,
            normalized_preds,
            labels=[label],
            average="macro",
            zero_division=0,
        )
    invalid_count = sum(1 for pred in predictions if pred is None)
    total = len(labels)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "invalid_rate": float(invalid_count / total) if total else 0.0,
        "invalid_count": int(invalid_count),
        "num_samples": int(total),
        "per_class_f1": {key: float(value) for key, value in per_class_f1.items()},
        "label_names": label_names,
    }
