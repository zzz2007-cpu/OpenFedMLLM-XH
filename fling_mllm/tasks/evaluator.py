import json
import os
from typing import Dict, Optional

from .registry import get_evaluator_cls, normalize_task_type
from ..utils.eval_utils import save_confusion_matrix_plot


def build_task_evaluator(
    task_type: str,
    eval_data_path: str,
    data_format: str = "auto",
    split: str = "eval",
    loader_kwargs: Optional[Dict] = None,
):
    evaluator_cls = get_evaluator_cls(normalize_task_type(task_type))
    return evaluator_cls(
        eval_data_path=eval_data_path,
        data_format=data_format,
        split=split,
        loader_kwargs=loader_kwargs,
    )


def run_task_evaluation(
    task_type: str,
    model,
    tokenizer,
    eval_data_path: str,
    output_dir: str,
    max_new_tokens: int = 16,
    max_samples: Optional[int] = None,
    sample_seed: int = 42,
    device: str = "cuda",
    data_format: str = "auto",
    split: str = "eval",
    loader_kwargs: Optional[Dict] = None,
    stage_tag: str = "Eval",
):
    evaluator = build_task_evaluator(
        task_type=task_type,
        eval_data_path=eval_data_path,
        data_format=data_format,
        split=split,
        loader_kwargs=loader_kwargs,
    )
    samples = evaluator.sample_eval_subset(
        max_samples=max_samples,
        sample_seed=sample_seed,
        force_full_eval=(max_samples is None),
    )
    metrics, records = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        max_new_tokens=max_new_tokens,
        device=device,
        stage_tag=stage_tag,
    )
    save_task_eval_outputs(output_dir=output_dir, metrics=metrics, records=records)
    return metrics, records


def save_task_eval_outputs(output_dir: str, metrics: Dict, records) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "eval_results.json")
    metrics_path = os.path.join(output_dir, "eval_metrics.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = metrics.get("confusion_matrix")
    label_names = metrics.get("label_names")
    if isinstance(cm, list) and isinstance(label_names, list) and label_names:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        save_confusion_matrix_plot(cm=cm, label_names=label_names, output_path=cm_path)
