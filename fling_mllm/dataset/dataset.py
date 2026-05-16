import inspect
from functools import partial
from typing import Dict, Optional

from torchvision import transforms

from fling_llm.dataset.utils import SupervisedDataset, data_collator

from ..adapters import MODEL_FAMILY_QWEN2_VL, resolve_model_family
from ..tasks import load_task_samples
from .qwen2_vl_dataset import build_qwen2_vl_data_module


def build_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def make_supervised_data_module(
    tokenizer,
    data_path,
    llm_type="minicpm",
    processor=None,
    model_name_or_path=None,
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
    enable_audio=False,
    bad_sample_log_path=None,
    task_type: str = "classification",
    data_format: str = "auto",
    split: str = "train",
    task_loader_kwargs: Optional[Dict] = None,
) -> Dict:
    loader_kwargs = dict(task_loader_kwargs or {})
    loader_kwargs.setdefault("require_answer", True)
    train_json = load_task_samples(
        data_path=data_path,
        task_type=task_type,
        split=split,
        data_format=data_format,
        **loader_kwargs,
    )

    model_family = resolve_model_family(model_name_or_path, llm_type=llm_type)
    if model_family == MODEL_FAMILY_QWEN2_VL:
        if processor is None:
            raise ValueError(
                "Qwen2-VL data module requires processor, but got None. "
                "Please pass model.processor from model_builder."
            )
        return build_qwen2_vl_data_module(
            train_json=train_json,
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            bad_sample_log_path=bad_sample_log_path,
        )

    transform = build_transform()
    dataset_kwargs = dict(
        slice_config=slice_config,
        llm_type=llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )
    signature = inspect.signature(SupervisedDataset.__init__)
    if "enable_audio" in signature.parameters:
        dataset_kwargs["enable_audio"] = enable_audio
    if "bad_sample_log_path" in signature.parameters:
        dataset_kwargs["bad_sample_log_path"] = bad_sample_log_path
    train_dataset = SupervisedDataset(
        train_json,
        transform,
        tokenizer,
        **dataset_kwargs,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": partial(data_collator, max_length=max_length),
    }
