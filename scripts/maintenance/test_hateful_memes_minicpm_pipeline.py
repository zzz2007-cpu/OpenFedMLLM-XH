#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_hateful_adapter_without_top_level_import():
    # The repo's top-level fling_mllm import requires transformers. This smoke
    # test keeps the default path data-only, so it loads the task adapter without
    # importing the full training stack.
    pkg = types.ModuleType("fling_mllm")
    pkg.__path__ = [str(PROJECT_ROOT / "fling_mllm")]
    tasks_pkg = types.ModuleType("fling_mllm.tasks")
    tasks_pkg.__path__ = [str(PROJECT_ROOT / "fling_mllm" / "tasks")]
    hateful_pkg = types.ModuleType("fling_mllm.tasks.hateful_memes")
    hateful_pkg.__path__ = [str(PROJECT_ROOT / "fling_mllm" / "tasks" / "hateful_memes")]
    sys.modules.setdefault("fling_mllm", pkg)
    sys.modules.setdefault("fling_mllm.tasks", tasks_pkg)
    sys.modules.setdefault("fling_mllm.tasks.hateful_memes", hateful_pkg)

    registry_name = "fling_mllm.tasks.registry"
    if registry_name not in sys.modules:
        registry_spec = importlib.util.spec_from_file_location(
            registry_name,
            PROJECT_ROOT / "fling_mllm" / "tasks" / "registry.py",
        )
        registry_mod = importlib.util.module_from_spec(registry_spec)
        sys.modules[registry_name] = registry_mod
        assert registry_spec is not None and registry_spec.loader is not None
        registry_spec.loader.exec_module(registry_mod)

    adapter_name = "fling_mllm.tasks.hateful_memes.data_adapter"
    adapter_spec = importlib.util.spec_from_file_location(
        adapter_name,
        PROJECT_ROOT / "fling_mllm" / "tasks" / "hateful_memes" / "data_adapter.py",
    )
    adapter_mod = importlib.util.module_from_spec(adapter_spec)
    sys.modules[adapter_name] = adapter_mod
    assert adapter_spec is not None and adapter_spec.loader is not None
    adapter_spec.loader.exec_module(adapter_mod)
    return adapter_mod


_adapter = _load_hateful_adapter_without_top_level_import()
LABEL_VERBALIZER = _adapter.LABEL_VERBALIZER
PROMPT_TEMPLATE = _adapter.PROMPT_TEMPLATE
format_hateful_memes_sample = _adapter.format_hateful_memes_sample
infer_hateful_memes_root = _adapter.infer_hateful_memes_root
load_hateful_memes_samples = _adapter.load_hateful_memes_samples


def read_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No records found in {path}")


def read_jsonl_at(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_no == index:
                return json.loads(line)
    raise IndexError(f"No record at index {index} in {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test Hateful Memes MiniCPM data pipeline.")
    p.add_argument("--data_root", default="hateful_memes")
    p.add_argument("--stat_setting", default="iid")
    p.add_argument("--modal_setting", default="aligned")
    p.add_argument("--client_id", type=int, default=0)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--check_minicpm_preprocess", action="store_true")
    p.add_argument("--model_name_or_path", default="openbmb/MiniCPM-V-2_6-int4")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    client_path = (
        data_root
        / "federated"
        / args.stat_setting
        / args.modal_setting
        / f"client_{args.client_id}.jsonl"
    )
    if not client_path.exists():
        raise FileNotFoundError(
            f"Client split not found: {client_path}. "
            "Check --stat_setting, --modal_setting, and --client_id."
        )

    raw_train = read_first_jsonl(data_root / "train.jsonl")
    inferred_root = infer_hateful_memes_root(str(client_path), explicit_root=str(data_root))
    formatted_raw = format_hateful_memes_sample(raw_train, root=inferred_root)
    client_raw = read_jsonl_at(client_path, args.sample_index)
    formatted_client = format_hateful_memes_sample(
        client_raw,
        root=inferred_root,
        require_answer=True,
        strict_image_path=True,
    )

    expected_answer = LABEL_VERBALIZER[int(raw_train["label"])]
    if formatted_raw["answer"] != expected_answer:
        raise AssertionError("Label verbalizer check failed.")
    if formatted_client.get("image") is not None and not Path(formatted_client["image"]).exists():
        raise AssertionError(f"Formatted image path does not exist: {formatted_client['image']}")

    print("=== Hateful Memes MiniCPM smoke test ===")
    print(f"data_root: {data_root}")
    print(f"client_path: {client_path}")
    print("loaded_client_samples: checked one selected JSONL record")
    print(f"prompt_template: {PROMPT_TEMPLATE!r}")
    print(f"label_verbalizer: {LABEL_VERBALIZER}")
    print("formatted_sample:")
    print(json.dumps(formatted_client, ensure_ascii=False, indent=2))

    if args.check_minicpm_preprocess:
        from functools import partial

        from fling_mllm.config.arguments import LoraArguments, ModelArguments, TrainingArguments
        from fling_mllm.dataset.dataset import make_supervised_data_module
        from fling_mllm.model.export_hf import export_hf_tokenizer
        from fling_mllm.utils.model_builder import build_model_and_tokenizer

        model_args = ModelArguments(model_name_or_path=args.model_name_or_path)
        training_args = TrainingArguments(
            output_dir=str(PROJECT_ROOT / "outputs" / "checkpoints" / "hateful_memes_smoke"),
            cache_dir=str(PROJECT_ROOT / "mllmzoo" / "cache"),
            per_device_train_batch_size=1,
            model_max_length=1024,
            max_slice_nums=9,
            vision_batch_size=13,
            llm_type="minicpm",
            use_lora=True,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
        )
        lora_args = LoraArguments(lora_r=8, lora_alpha=16, lora_dropout=0.05)
        model, tokenizer = build_model_and_tokenizer(model_args, training_args, lora_args)
        tokenizer = export_hf_tokenizer(tokenizer)
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_path=str(client_path),
            llm_type="minicpm",
            processor=getattr(model, "processor", None),
            model_name_or_path=args.model_name_or_path,
            slice_config=getattr(model.config, "slice_config", None),
            patch_size=getattr(model.config, "patch_size", 14),
            query_nums=getattr(model.config, "query_num", 64),
            batch_vision=getattr(model.config, "batch_vision_input", False),
            max_length=1024,
            task_type="hateful_memes",
            data_format="jsonl",
            task_loader_kwargs={
                "hateful_memes_root": str(data_root),
                "strict_image_path": False,
            },
        )
        example = data_module["train_dataset"][args.sample_index]
        batch = data_module["data_collator"]([example])
        print("minicpm_preprocess_batch_keys:", sorted(batch.keys()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
