from .registry import list_supported_task_types, normalize_task_type
from .data_loading import build_task_loader_kwargs, load_task_samples
from .evaluator import build_task_evaluator, run_task_evaluation, save_task_eval_outputs
from .path_resolver import resolve_client_data_path

# Trigger registry side-effects.
from .classification import data_adapter as _classification_data_adapter  # noqa: F401
from .classification import evaluator as _classification_evaluator  # noqa: F401
from .hateful_memes import data_adapter as _hateful_memes_data_adapter  # noqa: F401
from .hateful_memes import evaluator as _hateful_memes_evaluator  # noqa: F401
from .vqa import data_adapter as _vqa_data_adapter  # noqa: F401
from .vqa import evaluator as _vqa_evaluator  # noqa: F401

__all__ = [
    "normalize_task_type",
    "list_supported_task_types",
    "load_task_samples",
    "build_task_loader_kwargs",
    "build_task_evaluator",
    "run_task_evaluation",
    "save_task_eval_outputs",
    "resolve_client_data_path",
]
