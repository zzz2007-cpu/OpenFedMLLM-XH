from typing import Dict, List, Optional, Tuple

from ..base_evaluator import BaseTaskEvaluator
from ..registry import register_evaluator
from ...utils.eval_utils import (
    classify_prediction_error,
    compute_classification_metrics,
    extract_ground_truth,
    extract_question,
    generate_answer,
    generate_label_probability_scores,
    log_first_eval_sample_snapshot,
    match_prediction_to_label,
    summarize_prediction_errors,
)


@register_evaluator("classification")
class ClassificationTaskEvaluator(BaseTaskEvaluator):
    def __init__(
        self,
        eval_data_path: str,
        data_format: str = "auto",
        split: str = "eval",
        loader_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            eval_data_path=eval_data_path,
            task_type="classification",
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
        stage_tag: str = "Eval",
    ) -> Tuple[Dict, List[Dict]]:
        if score_max_new_tokens is None:
            score_max_new_tokens = max(64, int(max_new_tokens) * 4)

        if samples:
            log_first_eval_sample_snapshot(
                sample=samples[0],
                tokenizer=tokenizer,
                model=model,
                stage_tag=stage_tag,
            )

        label_set = sorted(
            set(extract_ground_truth(sample) for sample in samples if extract_ground_truth(sample))
        )

        predictions: List[str] = []
        ground_truths: List[str] = []
        score_matrix: List[Optional[List[float]]] = []
        records: List[Dict] = []

        for idx, sample in enumerate(samples):
            question = extract_question(sample)
            gt = extract_ground_truth(sample)
            image_path = sample.get("image")
            sample_id = sample.get("id", sample.get("question_id", idx))

            try:
                raw_pred = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    image_path=image_path,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            except Exception:
                raw_pred = ""

            matched_pred = match_prediction_to_label(raw_pred, label_set, question=question)
            label_scores = generate_label_probability_scores(
                model=model,
                tokenizer=tokenizer,
                question=question,
                label_set=label_set,
                image_path=image_path,
                max_new_tokens=score_max_new_tokens,
                device=device,
                fallback_text=raw_pred,
            )
            error_type = classify_prediction_error(
                raw_output=raw_pred,
                parsed_pred=matched_pred,
                ground_truth=gt,
            )

            predictions.append(matched_pred)
            ground_truths.append(gt)
            score_matrix.append(label_scores)
            records.append(
                {
                    "idx": idx,
                    "id": sample_id,
                    "image": image_path,
                    "question": question,
                    "ground_truth": gt,
                    "raw_prediction": raw_pred,
                    "matched_prediction": matched_pred,
                    "correct": bool(matched_pred == gt),
                    "error_type": error_type,
                }
            )

        metrics = compute_classification_metrics(
            predictions,
            ground_truths,
            label_set,
            score_matrix=score_matrix,
        )
        metrics["error_breakdown"] = summarize_prediction_errors(records)
        return metrics, records
