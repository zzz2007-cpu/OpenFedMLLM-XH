from typing import Dict, List, Optional

from .prompt_builder import build_vqa_prompt


def build_vqa_conversations(
    question: str,
    answer: str,
    prompt_template: Optional[str] = None,
) -> List[Dict[str, str]]:
    answer_text = "" if answer is None else str(answer).strip()
    if not answer_text:
        raise ValueError("VQA answer is empty; cannot build supervised conversation.")
    user_prompt = build_vqa_prompt(question=question, template=prompt_template)
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer_text},
    ]


def format_vqa_sample(
    sample_id: str,
    image_path: str,
    question: str,
    answer: str,
    answers: Optional[List[str]] = None,
    prompt_template: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    item = {
        "id": sample_id,
        "image": image_path,
        "question": question,
        "answer": answer,
        "answers": list(answers or []),
        "conversations": build_vqa_conversations(
            question=question,
            answer=answer,
            prompt_template=prompt_template,
        ),
    }
    if metadata:
        item.update(metadata)
    return item
