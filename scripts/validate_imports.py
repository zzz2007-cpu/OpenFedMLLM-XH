import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from fling_mllm import AutoModel
    from mllmzoo import list_models
    assert AutoModel is not None
    models = list_models()
    assert isinstance(models, list)
    print("import validation passed")


if __name__ == "__main__":
    main()
