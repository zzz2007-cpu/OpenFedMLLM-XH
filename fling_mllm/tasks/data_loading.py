from typing import Dict, List, Optional

from .registry import get_data_loader, normalize_task_type


def load_task_samples(
    data_path: str,
    task_type: str = "classification",
    split: str = "train",
    data_format: str = "auto",
    **kwargs,
) -> List[Dict]:
    task_key = normalize_task_type(task_type)
    loader = get_data_loader(task_key)
    return loader(
        data_path=data_path,
        split=split,
        data_format=data_format,
        **kwargs,
    )


def build_task_loader_kwargs(
    task_type: str,
    data_args=None,
    extra_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Build task-specific data loader kwargs from DataArguments.
    Unknown attributes are ignored.
    """
    kwargs = dict(extra_kwargs or {})
    task_key = normalize_task_type(task_type)

    if data_args is None:
        return kwargs

    def _get(name, default=None):
        return getattr(data_args, name, default)

    if task_key == "vqa":
        kwargs.setdefault("vqa_image_root", _get("vqa_image_root", None))
        kwargs.setdefault("vqa_prompt_template", _get("vqa_prompt_template", None))
        kwargs.setdefault("strict_image_path", _get("strict_image_path", True))
    if task_key == "hateful_memes":
        kwargs.setdefault("hateful_memes_root", _get("hateful_memes_root", None))
        kwargs.setdefault("strict_image_path", _get("strict_image_path", True))
    return kwargs
