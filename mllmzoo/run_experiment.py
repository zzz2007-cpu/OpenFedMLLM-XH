#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
from pathlib import Path

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def load_config_from_path(config_path: str):
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    spec = importlib.util.spec_from_file_location(f"mllmzoo_config_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_config_by_name(name: str):
    registry = _load_registry()
    model_registry = registry.MODEL_REGISTRY
    if name not in model_registry:
        raise KeyError(f"Unknown model config: {name}")
    config_path = Path(_PROJ_ROOT) / model_registry[name]["config_path"]
    return load_config_from_path(str(config_path))


def _load_registry():
    registry_path = Path(__file__).resolve().parent / "registry.py"
    spec = importlib.util.spec_from_file_location("mllmzoo_registry", registry_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLLM experiment with mode dispatch")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Registry config name from mllmzoo.registry")
    group.add_argument("--config", help="Path to config .py file")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["federated", "local_only", "centralized"],
        help="Optional override for run_args.mode in config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from fling_mllm.pipeline import run_mode_from_config

    module = load_config_by_name(args.name) if args.name else load_config_from_path(args.config)
    exp_args = module.exp_args
    run_mode_from_config(exp_args, mode_override=args.mode)


if __name__ == "__main__":
    main()
