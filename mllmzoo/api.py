import importlib.util
from pathlib import Path
from .registry import MODEL_REGISTRY
from .downloader import download_from_hf


def list_models():
    return sorted(MODEL_REGISTRY.keys())


def load_config(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model config: {name}")
    config_path = Path(__file__).resolve().parents[1] / MODEL_REGISTRY[name]["config_path"]
    spec = importlib.util.spec_from_file_location(f"mllmzoo_config_{name}", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def download_model(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return download_from_hf(MODEL_REGISTRY[name]["repo_id"], cache_subdir="models")


def _update_nested(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _update_nested(target[key], value)
        else:
            target[key] = value


def run_federated_finetune(name: str, hooks=None, overrides=None, mode_override=None):
    from fling_mllm.pipeline import run_mode_from_config
    module = load_config(name)
    exp_args = module.exp_args
    if overrides:
        _update_nested(exp_args, overrides)
    if hooks is not None:
        exp_args.hooks = hooks
    run_mode_from_config(exp_args, mode_override=mode_override)
