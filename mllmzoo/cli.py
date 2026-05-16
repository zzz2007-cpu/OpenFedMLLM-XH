import argparse
from .api import list_models, run_federated_finetune


def list_models_cli():
    for name in list_models():
        print(name)


def train_model_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", default=None, choices=["federated", "local_only", "centralized"])
    args = parser.parse_args()
    run_federated_finetune(args.name, mode_override=args.mode)
