"""Prompt utilities for ScienceQA multiple-choice evaluation and training."""

from __future__ import annotations

from typing import Any


def option_letter(index: int) -> str:
    if index < 0:
        raise ValueError("option index must be non-negative")
    return chr(ord("A") + int(index))


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


SCIENCEQA_TEXT_CONTEXT_KEYS = ("lecture",)


def build_text_context(sample: dict, include_text_context: bool = True) -> str:
    if not include_text_context:
        return ""
    pieces = []
    for key in SCIENCEQA_TEXT_CONTEXT_KEYS:
        value = normalize_text(sample.get(key))
        if value:
            pieces.append(value)
    seen = set()
    deduped = []
    for piece in pieces:
        if piece in seen:
            continue
        seen.add(piece)
        deduped.append(piece)
    return "\n".join(deduped).strip()


def build_scienceqa_prompt(
    sample: dict,
    include_text_context: bool | None = None,
) -> str:
    context = sample.get("available_context") or {}
    if include_text_context is None:
        include_text_context = bool(context.get("text_context", True))
    question = normalize_text(sample.get("question"))
    if not question:
        raise ValueError(f"ScienceQA sample {sample.get('id')} has empty question.")

    choices = sample.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"ScienceQA sample {sample.get('id')} has no choices.")

    lines = [
        f"Question: {question}",
        "Choices:",
    ]
    for idx, choice in enumerate(choices):
        lines.append(f"{option_letter(idx)}. {normalize_text(choice)}")

    text_context = build_text_context(sample, include_text_context=include_text_context)
    if text_context:
        lines.append(f"Context: {text_context}")

    lines.append("Please answer with the option letter only.")
    return "\n".join(lines).strip()


def answer_letter_from_sample(sample: dict) -> str:
    if sample.get("answer_letter"):
        return str(sample["answer_letter"]).strip().upper()
    return option_letter(int(sample["answer"]))


def image_path_for_context(sample: dict) -> str | None:
    context = sample.get("available_context") or {}
    if not bool(context.get("image", True)):
        return None
    image = sample.get("image")
    return str(image) if image else None
