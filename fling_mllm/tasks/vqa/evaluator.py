from typing import Dict, List, Optional, Tuple

from ..base_evaluator import BaseTaskEvaluator
from ..registry import register_evaluator
from ...utils.eval_utils import generate_answer, log_first_eval_sample_snapshot
from .metrics import compute_vqa_metrics, normalize_vqa_answer, score_vqa_prediction


def _question_from_sample(sample: Dict) -> str:
    question = sample.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    conversations = sample.get("conversations", [])
    for turn in conversations:
        if str(turn.get("role", "")).strip().lower() == "user":
            content = str(turn.get("content", "")).strip()
            if content:
                return content
    return ""


def _references_from_sample(sample: Dict) -> List[str]:
    refs = sample.get("answers")
    out = []
    if isinstance(refs, list):
        for x in refs:
            text = "" if x is None else str(x).strip()
            if text:
                out.append(text)
    elif isinstance(refs, str) and refs.strip():
        out.append(refs.strip())

    answer = sample.get("answer")
    if (not out) and isinstance(answer, str) and answer.strip():
        out.append(answer.strip())
    return out


@register_evaluator("vqa")
class VQATaskEvaluator(BaseTaskEvaluator):
    def __init__(
        self,
        eval_data_path: str,
        data_format: str = "auto",
        split: str = "eval",
        loader_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            eval_data_path=eval_data_path,
            task_type="vqa",
            data_format=data_format,
            split=split,
            loader_kwargs=loader_kwargs,
        )

    def evaluate(
        self,
        model,
        tokenizer,
        samples: List[Dict],
        max_new_tokens: int = 16,
        device: str = "cuda",
        score_max_new_tokens: Optional[int] = None,
        stage_tag: str = "VQAEval",
    ) -> Tuple[Dict, List[Dict]]:
        if samples and isinstance(samples[0], dict) and samples[0].get("conversations"):
            log_first_eval_sample_snapshot(
                sample=samples[0],
                tokenizer=tokenizer,
                model=model,
                stage_tag=stage_tag,
            )

        predictions: List[str] = []
        references_list: List[List[str]] = []
        answer_types: List[str] = []
        records: List[Dict] = []

        for idx, sample in enumerate(samples):
            question = _question_from_sample(sample)
            refs = _references_from_sample(sample)
            image_path = sample.get("image")
            sample_id = sample.get("id", sample.get("question_id", idx))
            answer_type = sample.get("answer_type", "unknown")

            try:
                pred = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    image_path=image_path,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            except Exception:
                pred = ""

            prediction = "" if pred is None else str(pred).strip()
            scores = score_vqa_prediction(prediction, refs)

            predictions.append(prediction)
            references_list.append(refs)
            answer_types.append(str(answer_type) if answer_type is not None else "unknown")

            records.append(
                {
                    "idx": idx,
                    "id": sample_id,
                    "image": image_path,
                    "question": question,
                    "ground_truth": sample.get("answer", refs[0] if refs else ""),
                    "references": refs,
                    "raw_prediction": prediction,
                    "prediction_normalized": normalize_vqa_answer(prediction),
                    "normalized_exact_match": scores["normalized_exact_match"],
                    "vqa_score": scores["vqa_score"],
                    "answer_type": answer_type,
                    "empty_prediction": (prediction == ""),
                }
            )

        metrics = compute_vqa_metrics(
            predictions=predictions,
            references_list=references_list,
            answer_types=answer_types,
        )
        metrics["task_type"] = "vqa"
        return metrics, records
