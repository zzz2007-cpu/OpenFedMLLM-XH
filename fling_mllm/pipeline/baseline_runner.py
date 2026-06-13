import copy
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..client.trainer.sft_mllm_fedavg_trainer import CPMTrainer
from ..dataset.dataset import make_supervised_data_module
from ..federated.hooks import FederatedHook
from ..tasks import (
    build_task_evaluator,
    build_task_loader_kwargs,
    load_task_samples,
    normalize_task_type,
    resolve_client_data_path,
    save_task_eval_outputs,
)
from ..utils.model_builder import build_model_and_tokenizer, get_parameter_number
from .generic_model_mllm_pipeline import (
    _audit_trainable_parameters,
    _build_slice_config,
    _sync_vision_batch_size,
)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_plain_dict(section) -> dict:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    try:
        return dict(section)
    except Exception:
        return {}


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _build_eval_loader_kwargs(eval_args: dict, task_type: str) -> dict:
    loader_kwargs = {}
    if task_type == "vqa":
        if eval_args.get("vqa_image_root") is not None:
            loader_kwargs["vqa_image_root"] = eval_args.get("vqa_image_root")
        if eval_args.get("vqa_prompt_template") is not None:
            loader_kwargs["vqa_prompt_template"] = eval_args.get("vqa_prompt_template")
        if eval_args.get("strict_image_path") is not None:
            loader_kwargs["strict_image_path"] = eval_args.get("strict_image_path")
    if task_type == "hateful_memes":
        if eval_args.get("hateful_memes_root") is not None:
            loader_kwargs["hateful_memes_root"] = eval_args.get("hateful_memes_root")
        if eval_args.get("strict_image_path") is not None:
            loader_kwargs["strict_image_path"] = eval_args.get("strict_image_path")
    return loader_kwargs


def _load_per_round_metrics(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[FinalModelEvalHook] WARNING: ignoring invalid per-round metric "
                    f"at {path}:{line_no}: {exc}"
                )
    return records


def _metric_value(record: dict, *keys: str) -> float:
    for key in keys:
        value = record.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float("-inf")


def _collect_client_data_paths(data_path: str, num_clients: int) -> List[str]:
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path does not exist: {data_path!r}")
    paths = []
    missing = []
    for client_idx in range(num_clients):
        try:
            fp = resolve_client_data_path(data_path, client_idx)
            paths.append(fp)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError(
            "Missing client shard files:\n" + "\n".join(missing[:20])
        )
    return paths


def _build_data_module(
    model,
    tokenizer,
    training_args,
    data_args,
    llm_type: str,
    data_path: str,
    bad_sample_log_path: str,
):
    slice_config = _build_slice_config(model, training_args.max_slice_nums)
    _sync_vision_batch_size(
        model=model,
        max_slice_nums=training_args.max_slice_nums,
        configured_vbs=getattr(training_args, "vision_batch_size", None),
    )
    patch_size = getattr(model.config, "patch_size", 14)
    query_num = getattr(model.config, "query_num", 64)
    batch_vision = getattr(model.config, "batch_vision_input", False)
    task_type = normalize_task_type(getattr(data_args, "task_type", "classification"))
    data_format = getattr(data_args, "data_format", "auto")
    train_split = getattr(data_args, "train_split", "train")
    task_loader_kwargs = build_task_loader_kwargs(task_type=task_type, data_args=data_args)
    return make_supervised_data_module(
        tokenizer=tokenizer,
        data_path=data_path,
        llm_type=llm_type,
        processor=getattr(model, "processor", None),
        model_name_or_path=getattr(getattr(model, "config", None), "_name_or_path", None),
        slice_config=slice_config,
        patch_size=patch_size,
        query_nums=query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
        enable_audio=getattr(training_args, "enable_audio", False),
        bad_sample_log_path=bad_sample_log_path,
        task_type=task_type,
        data_format=data_format,
        split=train_split,
        task_loader_kwargs=task_loader_kwargs,
    )


def run_shared_eval(
    model,
    tokenizer,
    eval_data_path: str,
    output_dir: str,
    max_new_tokens: int = 16,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    task_type: str = "classification",
    data_format: str = "auto",
    split: str = "eval",
    loader_kwargs: Optional[Dict] = None,
    sample_seed: int = 42,
    stage_tag: str = "BaselineEval",
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    evaluator = build_task_evaluator(
        task_type=task_type,
        eval_data_path=eval_data_path,
        data_format=data_format,
        split=split,
        loader_kwargs=loader_kwargs,
    )
    samples = evaluator.sample_eval_subset(
        max_samples=max_samples,
        sample_seed=sample_seed,
        force_full_eval=(max_samples is None),
    )
    metrics, records = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        max_new_tokens=max_new_tokens,
        device=device,
        stage_tag=stage_tag,
    )
    save_task_eval_outputs(output_dir=output_dir, metrics=metrics, records=records)
    return metrics


class PerRoundModelEvalHook(FederatedHook):
    """Evaluate a fixed validation subset after selected aggregation rounds."""

    def __init__(self, eval_args: dict, output_dir: str):
        self._eval_args = _to_plain_dict(eval_args)
        self._output_dir = output_dir
        self._metrics_path = os.path.join(output_dir, "eval_metrics_per_round.jsonl")
        self._sample_ids_path = os.path.join(output_dir, "eval_sample_ids.json")
        self._model = None
        self._tokenizer = None
        self._evaluator = None
        self._samples = []
        self._enabled = False

    def on_train_start(self, context):
        if not _is_main_process():
            return
        self._model = context.get("model")
        self._tokenizer = context.get("tokenizer")
        eval_data_path = self._eval_args.get("eval_data_path")
        if not eval_data_path or not os.path.exists(eval_data_path):
            print(
                f"[PerRoundModelEvalHook] WARNING: eval_data_path not found: "
                f"{eval_data_path!r}. Per-round evaluation is disabled."
            )
            return
        if self._model is None or self._tokenizer is None:
            print(
                "[PerRoundModelEvalHook] WARNING: model/tokenizer unavailable. "
                "Per-round evaluation is disabled."
            )
            return

        task_type = normalize_task_type(self._eval_args.get("task_type", "classification"))
        self._evaluator = build_task_evaluator(
            task_type=task_type,
            eval_data_path=eval_data_path,
            data_format=self._eval_args.get("data_format", "auto"),
            split=self._eval_args.get("eval_split", "eval"),
            loader_kwargs=_build_eval_loader_kwargs(self._eval_args, task_type),
        )
        configured_max_samples = self._eval_args.get("max_samples", 100)
        max_samples = (
            None if configured_max_samples is None else int(configured_max_samples)
        )
        sample_seed = int(self._eval_args.get("eval_sample_seed", 42))
        self._samples = self._evaluator.sample_eval_subset(
            max_samples=max_samples,
            sample_seed=sample_seed,
            force_full_eval=False,
        )
        sample_ids = [sample.get("id", idx) for idx, sample in enumerate(self._samples)]
        _save_json(
            self._sample_ids_path,
            {
                "eval_data_path": eval_data_path,
                "sample_seed": sample_seed,
                "max_samples": max_samples,
                "num_samples": len(self._samples),
                "sample_ids": sample_ids,
            },
        )
        self._enabled = True
        print(
            f"[PerRoundModelEvalHook] Initialized with {len(self._samples)} fixed samples "
            f"(eval_freq={int(self._eval_args.get('eval_freq', 1))}, seed={sample_seed})."
        )

    def on_aggregate_end(self, context):
        if not self._enabled:
            return
        round_idx = int(context.get("round_idx", -1))
        eval_freq = int(self._eval_args.get("eval_freq", 1))
        if eval_freq <= 0 or (round_idx + 1) % eval_freq != 0:
            return

        stage_tag = f"Round {round_idx + 1} Eval"
        was_training = bool(getattr(self._model, "training", False))
        device = self._eval_args.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._model.eval()
        try:
            metrics, records = self._evaluator.evaluate(
                model=self._model,
                tokenizer=self._tokenizer,
                samples=self._samples,
                max_new_tokens=int(self._eval_args.get("max_new_tokens", 16)),
                device=device,
                stage_tag=stage_tag,
            )
        finally:
            if was_training:
                self._model.train()

        record = {
            "round_idx": round_idx,
            "eval_scope": "sampled",
            **metrics,
        }
        os.makedirs(self._output_dir, exist_ok=True)
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        predictions_path = os.path.join(self._output_dir, "output_predictions.jsonl")
        with open(predictions_path, "a", encoding="utf-8") as f:
            for prediction in records:
                output = dict(prediction)
                output["round_idx"] = round_idx
                output["eval_scope"] = "sampled"
                f.write(json.dumps(output, ensure_ascii=False) + "\n")

        macro_f1 = metrics.get("macro_f1", metrics.get("f1_macro"))
        print(
            f"[{stage_tag}] Acc={metrics.get('accuracy', 0.0):.4f} "
            f"Macro-F1={float(macro_f1 or 0.0):.4f} "
            f"(n={metrics.get('num_samples', len(self._samples))})"
        )


class FinalModelEvalHook(FederatedHook):
    """
    Federated-mode final evaluation hook that reuses the same baseline evaluator.
    """

    def __init__(self, eval_args: dict, output_dir: str):
        self._eval_args = _to_plain_dict(eval_args)
        self._output_dir = output_dir
        self._model = None
        self._tokenizer = None

    def on_train_start(self, context):
        if not _is_main_process():
            return
        self._model = context.get("model")
        self._tokenizer = context.get("tokenizer")

    def on_train_end(self, context):
        if not _is_main_process():
            return
        eval_data_path = self._eval_args.get("eval_data_path")
        if not eval_data_path:
            return
        if self._model is None or self._tokenizer is None:
            return
        if not os.path.exists(eval_data_path):
            print(f"[FinalModelEvalHook] WARNING: eval_data_path not found: {eval_data_path}")
            return
        eval_dir = os.path.join(self._output_dir, "eval")
        task_type = normalize_task_type(self._eval_args.get("task_type", "classification"))
        loader_kwargs = _build_eval_loader_kwargs(self._eval_args, task_type)
        force_full_eval = bool(self._eval_args.get("final_eval_full", False))
        final_max_samples = None if force_full_eval else self._eval_args.get("max_samples", None)
        eval_scope = "full" if force_full_eval else f"max_samples={final_max_samples}"
        print(
            f"[FinalModelEvalHook] Running final evaluation ({eval_scope}) -> {eval_dir}"
        )
        metrics = run_shared_eval(
            model=self._model,
            tokenizer=self._tokenizer,
            eval_data_path=eval_data_path,
            output_dir=eval_dir,
            max_new_tokens=int(self._eval_args.get("max_new_tokens", 16)),
            max_samples=final_max_samples,
            sample_seed=int(self._eval_args.get("eval_sample_seed", 42)),
            device=self._eval_args.get("device", "cuda"),
            task_type=task_type,
            data_format=self._eval_args.get("data_format", "auto"),
            split=self._eval_args.get("eval_split", "eval"),
            loader_kwargs=loader_kwargs,
            stage_tag="Final Eval",
        )
        root_metrics_path = os.path.join(self._output_dir, "eval_metrics.json")
        _save_json(root_metrics_path, metrics)
        per_round_metrics = _load_per_round_metrics(
            os.path.join(self._output_dir, "eval_metrics_per_round.jsonl")
        )
        best_accuracy_record = None
        best_macro_f1_record = None
        if per_round_metrics:
            best_accuracy_record = max(
                per_round_metrics,
                key=lambda item: _metric_value(item, "accuracy"),
            )
            best_macro_f1_record = max(
                per_round_metrics,
                key=lambda item: _metric_value(item, "macro_f1", "f1_macro"),
            )
        best_round = (
            int(best_accuracy_record["round_idx"]) + 1
            if best_accuracy_record is not None
            else None
        )
        best_accuracy = (
            best_accuracy_record.get("accuracy")
            if best_accuracy_record is not None
            else metrics.get("accuracy")
        )
        best_macro_f1 = (
            best_macro_f1_record.get("macro_f1", best_macro_f1_record.get("f1_macro"))
            if best_macro_f1_record is not None
            else metrics.get("macro_f1", metrics.get("f1_macro"))
        )
        final_summary = {
            "dataset": self._eval_args.get("dataset"),
            "model": self._eval_args.get("model"),
            "algorithm": self._eval_args.get("algorithm"),
            "setting": self._eval_args.get("setting"),
            "label_level": self._eval_args.get("label_level"),
            "modality_level": self._eval_args.get("modality_level"),
            "ablation": self._eval_args.get("ablation"),
            "seed": self._eval_args.get("seed"),
            "final_accuracy": metrics.get("accuracy"),
            "final_macro_f1": metrics.get("macro_f1", metrics.get("f1_macro")),
            "invalid_rate": metrics.get("invalid_rate"),
            "best_round": best_round,
            "best_accuracy": best_accuracy,
            "best_accuracy_round": best_round,
            "best_macro_f1": best_macro_f1,
            "best_macro_f1_round": (
                int(best_macro_f1_record["round_idx"]) + 1
                if best_macro_f1_record is not None
                else None
            ),
            "output_dir": self._output_dir,
            "checkpoint_path": None,
            "eval_metrics_path": root_metrics_path,
        }
        _save_json(os.path.join(self._output_dir, "final_summary.json"), final_summary)


def _metric_scalar_view(metrics: dict) -> dict:
    scalar_keys = [
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "auc_ovr_macro",
        "auc_ovr_weighted",
        "normalized_exact_match",
        "vqa_score",
        "empty_prediction_rate",
        "num_samples",
    ]
    out = {}
    for key in scalar_keys:
        value = metrics.get(key)
        out[key] = value
    return out


def _aggregate_client_metrics(per_client_metrics: List[dict]) -> dict:
    agg = {}
    metric_keys = [
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "auc_ovr_macro",
        "auc_ovr_weighted",
        "normalized_exact_match",
        "vqa_score",
    ]
    for metric_name in metric_keys:
        valid = []
        for item in per_client_metrics:
            value = item["eval_metrics"].get(metric_name)
            if isinstance(value, (float, int)):
                valid.append((item["client_idx"], float(value)))
        if not valid:
            agg[metric_name] = {
                "num_valid_clients": 0,
                "mean": None,
                "std": None,
                "best": None,
                "best_client": None,
                "worst": None,
                "worst_client": None,
            }
            continue
        values = np.array([v for _, v in valid], dtype=np.float64)
        best_client, best_value = max(valid, key=lambda x: x[1])
        worst_client, worst_value = min(valid, key=lambda x: x[1])
        agg[metric_name] = {
            "num_valid_clients": int(len(valid)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "best": float(best_value),
            "best_client": int(best_client),
            "worst": float(worst_value),
            "worst_client": int(worst_client),
        }
    return agg


def _prepare_eval_args(eval_args: Optional[dict], data_args) -> dict:
    eval_args = _to_plain_dict(eval_args)
    eval_data_path = eval_args.get("eval_data_path") or getattr(data_args, "eval_data_path", None)
    if not eval_data_path:
        raise ValueError("eval_data_path is required for baseline evaluation.")
    task_type = normalize_task_type(
        eval_args.get("task_type", getattr(data_args, "task_type", "classification"))
    )
    data_format = eval_args.get("data_format", getattr(data_args, "data_format", "auto"))
    eval_split = eval_args.get("eval_split", getattr(data_args, "eval_split", "eval"))
    loader_kwargs = build_task_loader_kwargs(task_type=task_type, data_args=data_args)
    if task_type == "vqa":
        if eval_args.get("vqa_image_root") is not None:
            loader_kwargs["vqa_image_root"] = eval_args.get("vqa_image_root")
        if eval_args.get("vqa_prompt_template") is not None:
            loader_kwargs["vqa_prompt_template"] = eval_args.get("vqa_prompt_template")
        if eval_args.get("strict_image_path") is not None:
            loader_kwargs["strict_image_path"] = eval_args.get("strict_image_path")
    return {
        "eval_data_path": eval_data_path,
        "max_new_tokens": int(eval_args.get("max_new_tokens", 16)),
        "max_samples": eval_args.get("max_samples", None),
        "sample_seed": int(eval_args.get("eval_sample_seed", 42)),
        "device": eval_args.get("device", "cuda"),
        "task_type": task_type,
        "data_format": data_format,
        "eval_split": eval_split,
        "loader_kwargs": loader_kwargs,
    }


def _cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_model_pair(model_args, training_args, lora_args):
    model, tokenizer = build_model_and_tokenizer(model_args, training_args, lora_args)
    if training_args.use_lora:
        _audit_trainable_parameters(model, lora_args=lora_args)
    else:
        print(get_parameter_number(model))
    return model, tokenizer


def run_local_only_baseline(
    model_args,
    data_args,
    training_args,
    lora_args,
    fed_args,
    eval_args: Optional[dict] = None,
    hooks=None,
):
    if hooks:
        print("[local_only] INFO: hooks are ignored in local_only baseline mode.")
    os.makedirs(training_args.cache_dir, exist_ok=True)
    os.makedirs(training_args.output_dir, exist_ok=True)
    eval_cfg = _prepare_eval_args(eval_args, data_args)

    client_paths = _collect_client_data_paths(data_args.data_path, fed_args.num_clients)
    base_seed = int(getattr(training_args, "seed", 42))
    base_data_seed = int(getattr(training_args, "data_seed", base_seed))
    llm_type = training_args.llm_type

    local_records = []
    local_losses = []
    for client_idx, client_data_path in enumerate(client_paths):
        client_dir = os.path.join(training_args.output_dir, f"client_{client_idx}")
        eval_dir = os.path.join(client_dir, "eval")
        checkpoint_dir = os.path.join(client_dir, "checkpoint-final")
        os.makedirs(client_dir, exist_ok=True)

        # Keep the same initialization for every client model.
        _set_global_seed(base_seed)
        client_train_args = copy.deepcopy(training_args)
        client_train_args.output_dir = client_dir
        client_train_args.seed = base_seed + client_idx
        client_train_args.data_seed = base_data_seed + client_idx

        model, tokenizer = _build_model_pair(model_args, client_train_args, lora_args)
        bad_sample_log_path = os.path.join(client_dir, f"bad_samples_client_{client_idx}.jsonl")
        data_module = _build_data_module(
            model=model,
            tokenizer=tokenizer,
            training_args=client_train_args,
            data_args=data_args,
            llm_type=llm_type,
            data_path=client_data_path,
            bad_sample_log_path=bad_sample_log_path,
        )
        train_dataset = data_module["train_dataset"]
        trainer = CPMTrainer(
            model=model,
            tokenizer=tokenizer,
            args=client_train_args,
            **data_module,
        )
        trainer.fed_round_idx = 0
        trainer.fed_client_idx = client_idx
        train_result = trainer.train()
        local_losses.append(train_result.training_loss)
        trainer.save_state()
        trainer.save_model(checkpoint_dir)

        eval_metrics = run_shared_eval(
            model=model,
            tokenizer=tokenizer,
            eval_data_path=eval_cfg["eval_data_path"],
            output_dir=eval_dir,
            max_new_tokens=eval_cfg["max_new_tokens"],
            max_samples=eval_cfg["max_samples"],
            sample_seed=eval_cfg["sample_seed"],
            device=eval_cfg["device"],
            task_type=eval_cfg["task_type"],
            data_format=eval_cfg["data_format"],
            split=eval_cfg["eval_split"],
            loader_kwargs=eval_cfg["loader_kwargs"],
        )
        local_records.append(
            {
                "client_idx": client_idx,
                "train_data_path": client_data_path,
                "num_train_samples": len(train_dataset),
                "training_loss": float(train_result.training_loss),
                "checkpoint_dir": checkpoint_dir,
                "eval_output_dir": eval_dir,
                "eval_metrics": _metric_scalar_view(eval_metrics),
            }
        )

        del trainer
        del model
        del tokenizer
        _cleanup_cuda()

    summary = {
        "mode": "local_only",
        "num_clients": fed_args.num_clients,
        "seed_for_shared_initialization": base_seed,
        "per_client": local_records,
        "aggregate": _aggregate_client_metrics(local_records),
        "artifacts": {
            "output_dir": training_args.output_dir,
            "client_eval_pattern": os.path.join(training_args.output_dir, "client_*/eval/eval_metrics.json"),
        },
    }
    _save_json(os.path.join(training_args.output_dir, "local_only_summary.json"), summary)
    np.save(
        os.path.join(training_args.output_dir, "local_only_training_loss.npy"),
        np.array(local_losses, dtype=np.float32),
    )
    return summary


def _merge_client_shards(
    client_paths: List[str],
    merged_path: str,
    task_type: str = "classification",
    data_format: str = "auto",
    split: str = "train",
    loader_kwargs: Optional[Dict] = None,
) -> Tuple[int, List[dict]]:
    merged = []
    shard_stats = []
    loader_kwargs = dict(loader_kwargs or {})
    loader_kwargs.setdefault("require_answer", True)
    for client_idx, client_path in enumerate(client_paths):
        data = load_task_samples(
            data_path=client_path,
            task_type=task_type,
            split=split,
            data_format=data_format,
            **loader_kwargs,
        )
        merged.extend(data)
        shard_stats.append(
            {
                "client_idx": client_idx,
                "data_path": client_path,
                "num_samples": len(data),
            }
        )
    _save_json(merged_path, merged)
    return len(merged), shard_stats


def run_centralized_baseline(
    model_args,
    data_args,
    training_args,
    lora_args,
    fed_args,
    eval_args: Optional[dict] = None,
    hooks=None,
):
    if hooks:
        print("[centralized] INFO: hooks are ignored in centralized baseline mode.")
    os.makedirs(training_args.cache_dir, exist_ok=True)
    os.makedirs(training_args.output_dir, exist_ok=True)
    eval_cfg = _prepare_eval_args(eval_args, data_args)

    client_paths = _collect_client_data_paths(data_args.data_path, fed_args.num_clients)
    merged_train_path = os.path.join(training_args.output_dir, "merged_train.json")
    task_type = normalize_task_type(getattr(data_args, "task_type", "classification"))
    train_data_format = getattr(data_args, "data_format", "auto")
    train_split = getattr(data_args, "train_split", "train")
    train_loader_kwargs = build_task_loader_kwargs(task_type=task_type, data_args=data_args)
    total_samples, shard_stats = _merge_client_shards(
        client_paths=client_paths,
        merged_path=merged_train_path,
        task_type=task_type,
        data_format=train_data_format,
        split=train_split,
        loader_kwargs=train_loader_kwargs,
    )

    base_seed = int(getattr(training_args, "seed", 42))
    _set_global_seed(base_seed)
    model, tokenizer = _build_model_pair(model_args, training_args, lora_args)
    llm_type = training_args.llm_type
    bad_sample_log_path = os.path.join(training_args.output_dir, "bad_samples_merged.jsonl")
    data_module = _build_data_module(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        data_args=data_args,
        llm_type=llm_type,
        data_path=merged_train_path,
        bad_sample_log_path=bad_sample_log_path,
    )

    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    train_result = trainer.train()
    checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_state()
    trainer.save_model(checkpoint_dir)

    eval_dir = os.path.join(training_args.output_dir, "eval")
    eval_metrics = run_shared_eval(
        model=model,
        tokenizer=tokenizer,
        eval_data_path=eval_cfg["eval_data_path"],
        output_dir=eval_dir,
        max_new_tokens=eval_cfg["max_new_tokens"],
        max_samples=eval_cfg["max_samples"],
        sample_seed=eval_cfg["sample_seed"],
        device=eval_cfg["device"],
        task_type=eval_cfg["task_type"],
        data_format=eval_cfg["data_format"],
        split=eval_cfg["eval_split"],
        loader_kwargs=eval_cfg["loader_kwargs"],
    )

    summary = {
        "mode": "centralized",
        "num_clients_merged": fed_args.num_clients,
        "merged_total_samples": total_samples,
        "merged_shard_stats": shard_stats,
        "training_loss": float(train_result.training_loss),
        "checkpoint_dir": checkpoint_dir,
        "eval_output_dir": eval_dir,
        "eval_metrics": _metric_scalar_view(eval_metrics),
        "artifacts": {
            "output_dir": training_args.output_dir,
            "merged_train_path": merged_train_path,
            "bad_sample_log_path": bad_sample_log_path,
        },
    }
    _save_json(os.path.join(training_args.output_dir, "centralized_summary.json"), summary)
    return summary
