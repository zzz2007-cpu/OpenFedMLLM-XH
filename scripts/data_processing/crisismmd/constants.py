from __future__ import annotations

from typing import Dict, List, Tuple


HUMANITARIAN_LABELS: List[str] = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
    "other_relevant_information",
    "not_humanitarian",
]

HUMANITARIAN_OPTIONS: List[Tuple[str, str]] = [
    (chr(ord("A") + idx), label) for idx, label in enumerate(HUMANITARIAN_LABELS)
]

LABEL_TO_LETTER: Dict[str, str] = {label: letter for letter, label in HUMANITARIAN_OPTIONS}
LETTER_TO_LABEL: Dict[str, str] = {letter: label for letter, label in HUMANITARIAN_OPTIONS}

DEFAULT_INPUT_ROOT = "crisismmd_datasplit_all"
DEFAULT_IMAGE_PREFIX = "../../data//crisis-mmd/raw_data"
DEFAULT_LABEL_SOURCE = "label_image"

DEFAULT_STANDARDIZED_DIRNAME = "humanitarian_standardized"

MANIFEST_FILENAME = "dataset_manifest.json"
STATS_FILENAME = "stats.json"
VALIDATION_FILENAME = "validation.json"

STANDARD_SPLIT_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
}

QUESTION_BIMODAL = "What is the humanitarian category based on the image and text?"
QUESTION_TEXT_ONLY = "What is the humanitarian category based on the text?"
QUESTION_IMAGE_ONLY = "What is the humanitarian category based on the image?"
QUESTION_NO_MODALITY = "What is the humanitarian category?"

INSTRUCTION_BIMODAL = (
    "Select the best answer to the following multiple-choice question based on the text and image."
)
INSTRUCTION_TEXT_ONLY = (
    "Select the best answer to the following multiple-choice question based on the text."
)
INSTRUCTION_IMAGE_ONLY = (
    "Select the best answer to the following multiple-choice question based on the image."
)
INSTRUCTION_NO_MODALITY = "Select the best answer to the following multiple-choice question."

ANSWER_SUFFIX = "The best answer is:"

PROMPT_MODE_TO_QUESTION = {
    "multimodal": QUESTION_BIMODAL,
    "text_only": QUESTION_TEXT_ONLY,
    "image_only": QUESTION_IMAGE_ONLY,
    "no_modality": QUESTION_NO_MODALITY,
}

PROMPT_QUESTIONS = set(PROMPT_MODE_TO_QUESTION.values())

SUPPORTED_MISSING_STRATEGIES = ("image", "text", "random")
SUPPORTED_CROSS_STRATEGIES = ("client_partition", "client_probability")

