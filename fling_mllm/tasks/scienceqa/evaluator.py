"""ScienceQA evaluator."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..base_evaluator import BaseTaskEvaluator
from ..registry import register_evaluator
from ...utils.eval_utils import generate_answer, log_first_eval_sample_snapshot
from .metrics import compute_scienceqa_metrics, parse_scienceqa_prediction
from .prompt_builder import answer_letter_from_sample, build_scienceqa_prompt, image_path_for_context


@register_evaluator("scienceqa")
class ScienceQATaskEvaluator(BaseTaskEvaluator):
    def __init__(
        self,
        eval_data_path: str,
        data_format: str = "auto",
        split: str = "eval",
        loader_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            eval_data_path=eval_data_path,
            task_type="scienceqa",
            data_format=data_format,
            split=split,
            loader_kwargs=loader_kwargs,
        )

    def evaluate(
        self,
        model,
        tokenizer,
        samples: List[Dict],
        max_new_tokens: int = 8,
        device: str = "cuda",
        score_max_new_tokens: Optional[int] = None,
        stage_tag: str = "ScienceQAEval",
    ) -> Tuple[Dict, List[Dict]]:
        del score_max_new_tokens
        if samples and model is not None and tokenizer is not None:
            log_first_eval_sample_snapshot(
                sample=samples[0],
                tokenizer=tokenizer,
                model=model,
                stage_tag=stage_tag,
            )

        predictions: List[str | None] = []
        labels: List[str] = []
        records: List[Dict] = []

        for idx, sample in enumerate(samples):
            prompt = sample.get("prompt") or build_scienceqa_prompt(sample)
            image_path = image_path_for_context(sample)
            label = answer_letter_from_sample(sample)
            raw_pred = ""
            if model is not None and tokenizer is not None:
                try:
                    raw_pred = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        question=prompt,
                        image_path=image_path,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        enforce_letter_output=True,
                    )
                except Exception:
                    raw_pred = ""
            parsed = parse_scienceqa_prediction(raw_pred, num_choices=len(sample.get("choices") or []))
            predictions.append(parsed)
            labels.append(label)
            records.append({
                "idx": idx,
                "id": sample.get("id", idx),
                "image": image_path,
                "question": sample.get("question_raw", sample.get("question", "")),
                "prompt": prompt,
                "choices": sample.get("choices"),
                "ground_truth": label,
                "raw_prediction": raw_pred,
                "parsed_prediction": parsed,
                "correct": bool(parsed == label),
                "invalid_prediction": parsed is None,
                "client_modality_type": sample.get("client_modality_type"),
                "available_context": sample.get("available_context"),
            })

        metrics = compute_scienceqa_metrics(predictions=predictions, labels=labels)
        metrics["task_type"] = "scienceqa"
        return metrics, records
