import os
import random
from typing import Dict, List, Optional

from .data_loading import load_task_samples
from .registry import normalize_task_type


class BaseTaskEvaluator:
    def __init__(
        self,
        eval_data_path: str,
        task_type: str,
        data_format: str = "auto",
        split: str = "eval",
        loader_kwargs: Optional[Dict] = None,
    ):
        self.eval_data_path = eval_data_path
        self.task_type = normalize_task_type(task_type)
        self.data_format = data_format
        self.split = split
        self.loader_kwargs = dict(loader_kwargs or {})

        if not self.eval_data_path:
            raise ValueError("eval_data_path is required.")
        if not os.path.exists(self.eval_data_path):
            raise FileNotFoundError(f"eval_data_path not found: {self.eval_data_path}")

        self._all_samples: Optional[List[Dict]] = None

    def load_all_samples(self) -> List[Dict]:
        if self._all_samples is None:
            self._all_samples = load_task_samples(
                data_path=self.eval_data_path,
                task_type=self.task_type,
                split=self.split,
                data_format=self.data_format,
                require_answer=False,
                **self.loader_kwargs,
            )
        return self._all_samples

    def sample_eval_subset(
        self,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
        force_full_eval: bool = False,
    ) -> List[Dict]:
        samples = list(self.load_all_samples())
        if force_full_eval or max_samples is None:
            return samples
        max_samples = int(max_samples)
        if max_samples <= 0 or len(samples) <= max_samples:
            return samples
        rng = random.Random(int(sample_seed))
        return rng.sample(samples, max_samples)

    def evaluate(
        self,
        model,
        tokenizer,
        samples: List[Dict],
        max_new_tokens: int = 16,
        device: str = "cuda",
        score_max_new_tokens: Optional[int] = None,
        stage_tag: str = "Eval",
    ):
        raise NotImplementedError
