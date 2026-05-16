import json
import os
from typing import Dict, List

from ..registry import register_data_loader


def _normalize_crisismmd_image_paths(samples: List[Dict], project_root: str) -> List[Dict]:
    """
    Keep backward-compatible image path correction for existing CrisisMMD shards.
    """
    real_image_root = os.path.join(project_root, "data", "crisis-mmd", "raw_data", "data_image")
    for item in samples:
        img = item.get("image")
        if not isinstance(img, str):
            continue
        if "data_image" not in img:
            continue
        suffix = img.split("data_image", 1)[-1].lstrip("/\\")
        item["image"] = os.path.join(real_image_root, suffix)
    return samples


@register_data_loader("classification")
def load_classification_samples(
    data_path: str,
    split: str = "train",
    data_format: str = "auto",
    **kwargs,
) -> List[Dict]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Classification data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(
            f"Classification data must be a JSON list, but got {type(payload).__name__}: {data_path}"
        )

    project_root = os.path.abspath(os.getcwd())
    samples = _normalize_crisismmd_image_paths(payload, project_root=project_root)

    for idx, sample in enumerate(samples):
        if "conversations" not in sample:
            raise ValueError(
                f"Invalid classification sample at idx={idx}: missing 'conversations' field. "
                f"Path={data_path}"
            )
    return samples
