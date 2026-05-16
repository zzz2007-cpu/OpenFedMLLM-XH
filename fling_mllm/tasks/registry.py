from typing import Callable, Dict, Type


_DATA_LOADER_REGISTRY: Dict[str, Callable] = {}
_EVALUATOR_REGISTRY: Dict[str, Type] = {}


def normalize_task_type(task_type: str) -> str:
    if task_type is None:
        return "classification"
    normalized = str(task_type).strip().lower()
    return normalized or "classification"


def register_data_loader(task_type: str):
    key = normalize_task_type(task_type)

    def decorator(fn: Callable):
        _DATA_LOADER_REGISTRY[key] = fn
        return fn

    return decorator


def register_evaluator(task_type: str):
    key = normalize_task_type(task_type)

    def decorator(cls):
        _EVALUATOR_REGISTRY[key] = cls
        return cls

    return decorator


def get_data_loader(task_type: str) -> Callable:
    key = normalize_task_type(task_type)
    if key not in _DATA_LOADER_REGISTRY:
        supported = sorted(_DATA_LOADER_REGISTRY.keys())
        raise KeyError(f"Unsupported task_type={task_type!r}. Supported data loaders: {supported}")
    return _DATA_LOADER_REGISTRY[key]


def get_evaluator_cls(task_type: str):
    key = normalize_task_type(task_type)
    if key not in _EVALUATOR_REGISTRY:
        supported = sorted(_EVALUATOR_REGISTRY.keys())
        raise KeyError(f"Unsupported task_type={task_type!r}. Supported evaluators: {supported}")
    return _EVALUATOR_REGISTRY[key]


def list_supported_task_types():
    task_keys = set(_DATA_LOADER_REGISTRY.keys()) | set(_EVALUATOR_REGISTRY.keys())
    return sorted(task_keys)
