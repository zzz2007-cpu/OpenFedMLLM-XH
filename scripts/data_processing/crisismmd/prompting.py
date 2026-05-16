from __future__ import annotations

from typing import Dict, List

from .constants import (
    ANSWER_SUFFIX,
    HUMANITARIAN_OPTIONS,
    INSTRUCTION_BIMODAL,
    INSTRUCTION_IMAGE_ONLY,
    INSTRUCTION_NO_MODALITY,
    INSTRUCTION_TEXT_ONLY,
    LABEL_TO_LETTER,
    PROMPT_MODE_TO_QUESTION,
)


def options_block() -> str:
    return "\n".join(f"({letter}) {label}" for letter, label in HUMANITARIAN_OPTIONS)


def modality_pattern_name(has_image: bool, has_text: bool) -> str:
    if has_image and has_text:
        return "multimodal"
    if has_image:
        return "image_only"
    if has_text:
        return "text_only"
    return "no_modality"


def build_user_prompt(text: str, has_image: bool, has_text: bool) -> str:
    mode = modality_pattern_name(has_image=has_image, has_text=has_text)
    prefix = "<image>\n" if has_image else ""
    options = options_block()

    if mode == "multimodal":
        instruction = INSTRUCTION_BIMODAL
        body_lines = [text.strip()]
    elif mode == "text_only":
        instruction = INSTRUCTION_TEXT_ONLY
        body_lines = [text.strip()]
    elif mode == "image_only":
        instruction = INSTRUCTION_IMAGE_ONLY
        body_lines = []
    else:
        instruction = INSTRUCTION_NO_MODALITY
        body_lines = []

    lines: List[str] = [prefix + instruction if prefix else instruction]
    for line in body_lines:
        if line:
            lines.append(line)
    lines.append(PROMPT_MODE_TO_QUESTION[mode])
    lines.append("Options:")
    lines.extend(options.splitlines())
    lines.append(ANSWER_SUFFIX)
    return "\n".join(lines)


def format_assistant_answer(label: str) -> str:
    if label not in LABEL_TO_LETTER:
        raise ValueError(f"Unknown humanitarian label: {label}")
    return f"({LABEL_TO_LETTER[label]}) {label}"


def build_conversations(text: str, label: str, has_image: bool, has_text: bool) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_user_prompt(text=text, has_image=has_image, has_text=has_text),
        },
        {
            "role": "assistant",
            "content": format_assistant_answer(label),
        },
    ]
