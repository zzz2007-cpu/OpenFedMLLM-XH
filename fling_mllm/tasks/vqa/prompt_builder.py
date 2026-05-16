from typing import Optional


DEFAULT_VQA_PROMPT_TEMPLATE = (
    "<image>\n"
    "Answer the following visual question using the image. "
    "Keep the answer short and precise, without extra explanation.\n"
    "Question: {question}\n"
    "Answer:"
)


def normalize_question_text(question: Optional[str]) -> str:
    text = "" if question is None else str(question)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    normalized = " ".join(lines).strip()
    return normalized


def build_vqa_prompt(
    question: str,
    template: Optional[str] = None,
) -> str:
    normalized_question = normalize_question_text(question)
    if not normalized_question:
        raise ValueError("VQA question is empty after normalization.")
    template = template or DEFAULT_VQA_PROMPT_TEMPLATE
    return template.format(question=normalized_question)
