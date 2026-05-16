from typing import Dict, Optional

from ..classification.evaluator import ClassificationTaskEvaluator
from ..registry import register_evaluator


@register_evaluator("hateful_memes")
class HatefulMemesTaskEvaluator(ClassificationTaskEvaluator):
    def __init__(
        self,
        eval_data_path: str,
        data_format: str = "auto",
        split: str = "eval",
        loader_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            eval_data_path=eval_data_path,
            data_format=data_format,
            split=split,
            loader_kwargs=loader_kwargs,
        )
        self.task_type = "hateful_memes"
