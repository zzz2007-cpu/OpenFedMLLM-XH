import importlib.util
from pathlib import Path


def load_default_exp_args():
    default_config = Path(__file__).resolve().parent / "minicpm" / "minicpmv-crisismmid-FedAvg.py"
    spec = importlib.util.spec_from_file_location("mllmzoo_default_config", default_config)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.exp_args


__all__ = ["load_default_exp_args"]

