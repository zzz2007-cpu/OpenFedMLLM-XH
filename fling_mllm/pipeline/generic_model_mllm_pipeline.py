import copy
import math
import os
from collections import defaultdict
import numpy as np
import transformers
from tqdm import tqdm
from PIL import Image
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from ..config.arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments, FedArguments
from ..model.export_hf import export_hf_tokenizer
from ..dataset.dataset import make_supervised_data_module
from ..client.trainer.sft_mllm_fedavg_trainer import CPMTrainer
from ..client.trainer.sft_mllm_fedprox_trainer import CPMTrainerReg
from ..client.trainer.sft_mllm_scaffold_trainer import CPMTrainerScaffold
from ..client.trainer.sft_mllm_fednova_trainer import CPMTrainerFedNova
from ..federated.aggregation import get_clients_this_round, global_aggregate
from ..federated.state import get_proxy_dict, get_auxiliary_dict
from ..federated.hooks import FederatedHookManager, summarize_state_dict
from ..tasks import (
    build_task_loader_kwargs,
    normalize_task_type,
    resolve_client_data_path,
)
from ..utils.model_builder import build_model_and_tokenizer, get_parameter_number


def _resolve_outer_lr_warmup_rounds(total_rounds, warmup_rounds=None):
    total_rounds = max(1, int(total_rounds))
    if warmup_rounds is None:
        warmup_rounds = max(1, int(total_rounds * 0.01))
    warmup_rounds = max(0, int(warmup_rounds))
    if total_rounds <= 1:
        return 0
    return min(warmup_rounds, total_rounds - 1)


def compute_outer_learning_rate(round_idx, base_lr, total_rounds, schedule="cosine", warmup_rounds=None, eta_min=0.0):
    schedule = str(schedule or "cosine").strip().lower()
    base_lr = float(base_lr)
    eta_min = float(eta_min)
    total_rounds = max(1, int(total_rounds))
    round_idx = max(0, int(round_idx))
    if eta_min > base_lr:
        raise ValueError(f"outer_lr_eta_min ({eta_min}) cannot be greater than init_learning_rate ({base_lr}).")
    if schedule in {"constant", "const"}:
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported outer_lr_schedule: {schedule}")

    warmup_rounds = _resolve_outer_lr_warmup_rounds(total_rounds, warmup_rounds)
    if round_idx < warmup_rounds:
        return base_lr * (round_idx + 1) / max(1, warmup_rounds)

    decay_span = max(1, total_rounds - warmup_rounds)
    return eta_min + (base_lr - eta_min) * (
        1 + math.cos(math.pi * (round_idx - warmup_rounds) / decay_span)
    ) / 2.0


def _alg_match(fed_alg, target):
    fed_alg = str(fed_alg).lower()
    return fed_alg == target or target in fed_alg


def _build_slice_config(model, max_slice_nums):
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = max_slice_nums
        return model.config.slice_config.to_dict()
    model.config.max_slice_nums = max_slice_nums
    return model.config.to_dict()


def _sync_vision_batch_size(model, max_slice_nums, configured_vbs=None):
    # Derive a safe default from max_slice_nums so slicing and vision chunking
    # stay coherent under one configuration source.
    base = int(max_slice_nums) + 4
    target = int(configured_vbs) if configured_vbs is not None else base
    current = int(getattr(model.config, "vision_batch_size", 1))
    final_vbs = max(current, target)
    model.config.vision_batch_size = final_vbs
    return final_vbs


def _state_delta_l2(local_state, global_state):
    delta = 0.0
    for key, value in local_state.items():
        if key not in global_state:
            continue
        diff = (value.float() - global_state[key].float()).norm().item()
        delta += diff ** 2
    return delta ** 0.5


def _is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_finite_number(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


_TRAIN_SAMPLE_SNAPSHOT_PRINTED = False


def _truncate_for_log(text, max_chars=1200):
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def _extract_turn_text(conversations, role):
    for turn in (conversations or []):
        if turn.get("role") == role:
            return str(turn.get("content", ""))
    return ""


def _extract_last_turn_text(conversations, role):
    for turn in reversed(conversations or []):
        if turn.get("role") == role:
            return str(turn.get("content", ""))
    return ""


def _safe_decode(tokenizer, token_ids):
    if token_ids is None:
        return ""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        return f"<decode_failed: {type(exc).__name__}: {exc}>"


def _first_non_ignore_pos(label_tensor):
    if label_tensor is None:
        return -1
    try:
        nz = label_tensor.ne(-100).nonzero(as_tuple=False)
        if nz.numel() == 0:
            return -1
        return int(nz[0].item())
    except Exception:
        return -1


def _summarize_pixel_values(pixel_values):
    if pixel_values is None:
        return {"present": False}
    if isinstance(pixel_values, list):
        shapes = []
        for idx, pv in enumerate(pixel_values):
            shape = tuple(pv.shape) if hasattr(pv, "shape") else type(pv).__name__
            shapes.append({"index": idx, "shape": shape})
        return {"present": True, "type": "list", "num_items": len(pixel_values), "items": shapes}
    shape = tuple(pixel_values.shape) if hasattr(pixel_values, "shape") else type(pixel_values).__name__
    return {"present": True, "type": type(pixel_values).__name__, "shape": shape}


def _summarize_image_loading(image_field):
    summary = {"raw_image_field_type": type(image_field).__name__, "loaded": False, "details": []}
    entries = []
    if isinstance(image_field, str):
        entries = [("<image>", image_field)]
    elif isinstance(image_field, dict):
        entries = list(image_field.items())
    else:
        summary["details"].append({"warning": f"unsupported image field type: {type(image_field).__name__}"})
        return summary

    all_ok = True
    for key, path in entries:
        item = {"slot": key, "path": path, "exists": os.path.exists(path)}
        if not item["exists"]:
            all_ok = False
            item["warning"] = "image path does not exist"
            summary["details"].append(item)
            continue
        try:
            with Image.open(path) as img:
                item["pil_mode"] = img.mode
                item["pil_size"] = tuple(img.size)
        except Exception as exc:
            all_ok = False
            item["warning"] = f"image load failed: {type(exc).__name__}: {exc}"
        summary["details"].append(item)
    summary["loaded"] = all_ok and len(entries) > 0
    return summary


def _print_first_training_sample_snapshot(local_datasets, tokenizer):
    global _TRAIN_SAMPLE_SNAPSHOT_PRINTED
    if _TRAIN_SAMPLE_SNAPSHOT_PRINTED:
        return
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(
        f"[TrainSampleDebug] Snapshot hook triggered (rank={rank}, world_size={world_size}).",
        flush=True,
    )
    if not local_datasets:
        print("[TrainSampleDebug] WARNING: no local datasets available.", flush=True)
        _TRAIN_SAMPLE_SNAPSHOT_PRINTED = True
        return

    dm0 = local_datasets[0]
    train_dataset = dm0.get("train_dataset")
    collator = dm0.get("data_collator")
    if train_dataset is None or len(train_dataset) == 0:
        print("[TrainSampleDebug] WARNING: train dataset is empty.", flush=True)
        _TRAIN_SAMPLE_SNAPSHOT_PRINTED = True
        return

    raw_sample = train_dataset.raw_data[0] if hasattr(train_dataset, "raw_data") else {}
    raw_sample_id = raw_sample.get("id", "N/A")
    raw_image_field = raw_sample.get("image")
    raw_conversations = raw_sample.get("conversations", [])
    raw_user = _extract_turn_text(raw_conversations, "user")
    raw_assistant = _extract_last_turn_text(raw_conversations, "assistant")
    image_status = _summarize_image_loading(raw_image_field)

    try:
        processed_sample = train_dataset[0]
        process_error = None
    except Exception as exc:
        processed_sample = None
        process_error = f"{type(exc).__name__}: {exc}"

    print("\n" + "=" * 90, flush=True)
    print("[TrainSampleDebug] First training sample snapshot (printed once before round-1).", flush=True)
    print("[TrainSampleDebug] Stage 1: Raw JSON sample", flush=True)
    print(f"  - id: {raw_sample_id}", flush=True)
    print(f"  - image_field: {_truncate_for_log(raw_image_field, 800)}", flush=True)
    print(f"  - image_load_status: {image_status}", flush=True)
    print(f"  - user_text: {repr(_truncate_for_log(raw_user, 1800))}", flush=True)
    print(f"  - assistant_text: {repr(_truncate_for_log(raw_assistant, 1800))}", flush=True)

    if processed_sample is None:
        print("[TrainSampleDebug] Stage 2/3 skipped due to preprocessing failure.", flush=True)
        print(f"  - preprocess_error: {process_error}", flush=True)
        print("=" * 90 + "\n", flush=True)
        _TRAIN_SAMPLE_SNAPSHOT_PRINTED = True
        return

    input_ids = processed_sample.get("input_ids")
    labels = processed_sample.get("labels")
    pre_input_decode = _safe_decode(tokenizer, input_ids)
    label_ids = []
    if labels is not None:
        label_ids = labels[labels != -100].long().tolist()
    pre_label_decode = _safe_decode(tokenizer, label_ids)

    print("[TrainSampleDebug] Stage 2: Preprocessed single sample (__getitem__ output)", flush=True)
    print(f"  - input_ids_len: {int(input_ids.numel()) if input_ids is not None else 0}", flush=True)
    print(f"  - input_ids_decode: {repr(_truncate_for_log(pre_input_decode, 1800))}", flush=True)
    print(f"  - labels_non_-100_token_count: {len(label_ids)}", flush=True)
    print(f"  - labels_non_-100_head_ids[:16]: {label_ids[:16]}", flush=True)
    print(f"  - labels_non_-100_decode: {repr(_truncate_for_log(pre_label_decode, 1800))}", flush=True)
    first_label_pos = _first_non_ignore_pos(labels)
    print(f"  - labels_first_non_-100_pos: {first_label_pos}", flush=True)
    if first_label_pos >= 0 and input_ids is not None:
        left = max(0, first_label_pos - 2)
        right = min(int(input_ids.numel()), first_label_pos + 4)
        win_ids = input_ids[left:right]
        win_decode = _safe_decode(tokenizer, win_ids)
        print(
            f"  - labels_first_pos_input_window[{left}:{right}]: "
            f"{repr(_truncate_for_log(win_decode, 400))}",
            flush=True,
        )
    print(
        "  - labels_note: labels_non_-100_decode is unshifted target text; "
        "causal shift is applied in trainer.compute_loss.",
        flush=True,
    )
    print(f"  - pixel_values: {_summarize_pixel_values(processed_sample.get('pixel_values'))}", flush=True)
    print(f"  - tgt_sizes: {processed_sample.get('tgt_sizes')}", flush=True)
    print(f"  - image_bound: {processed_sample.get('image_bound')}", flush=True)

    if callable(collator):
        try:
            batch = collator([processed_sample])
        except Exception as exc:
            batch = None
            print(f"[TrainSampleDebug] WARNING: data_collator failed: {type(exc).__name__}: {exc}", flush=True)
    else:
        batch = None
        print("[TrainSampleDebug] WARNING: no callable data_collator found.", flush=True)

    if batch is not None:
        b_input = batch.get("input_ids")
        b_labels = batch.get("labels")
        b_decode = _safe_decode(tokenizer, b_input[0] if b_input is not None else None)
        b_label_ids = []
        if b_labels is not None:
            b_label_ids = b_labels[0][b_labels[0] != -100].long().tolist()
        b_label_decode = _safe_decode(tokenizer, b_label_ids)
        batch_shapes = {}
        for k, v in batch.items():
            if hasattr(v, "shape"):
                batch_shapes[k] = tuple(v.shape)
            elif isinstance(v, list):
                batch_shapes[k] = f"list(len={len(v)})"
            else:
                batch_shapes[k] = type(v).__name__

        print("[TrainSampleDebug] Stage 3: Final batch fed to model (after data_collator)", flush=True)
        print(f"  - batch_shapes: {batch_shapes}", flush=True)
        print(f"  - batch_input_ids_decode[0]: {repr(_truncate_for_log(b_decode, 1800))}", flush=True)
        print(f"  - batch_labels_non_-100_head_ids[0][:16]: {b_label_ids[:16]}", flush=True)
        print(f"  - batch_labels_non_-100_decode[0]: {repr(_truncate_for_log(b_label_decode, 1800))}", flush=True)
        b_first_label_pos = _first_non_ignore_pos(b_labels[0] if b_labels is not None else None)
        print(f"  - batch_labels_first_non_-100_pos[0]: {b_first_label_pos}", flush=True)
        if b_first_label_pos >= 0 and b_input is not None:
            left = max(0, b_first_label_pos - 2)
            right = min(int(b_input[0].numel()), b_first_label_pos + 4)
            b_win_ids = b_input[0][left:right]
            b_win_decode = _safe_decode(tokenizer, b_win_ids)
            print(
                f"  - batch_labels_first_pos_input_window[0][{left}:{right}]: "
                f"{repr(_truncate_for_log(b_win_decode, 400))}",
                flush=True,
            )
        if "pixel_values" in batch:
            print(f"  - batch_pixel_values: {_summarize_pixel_values(batch.get('pixel_values'))}", flush=True)
    print("=" * 90 + "\n", flush=True)
    _TRAIN_SAMPLE_SNAPSHOT_PRINTED = True


def _aggregate_trainable_by_prefix(trainable_items, depth):
    stats = defaultdict(int)
    for name, _, numel in trainable_items:
        parts = name.split(".")
        key = ".".join(parts[:depth]) if len(parts) >= depth else name
        stats[key] += numel
    return sorted(stats.items(), key=lambda x: x[1], reverse=True)


def _audit_trainable_parameters(model, lora_args=None):
    """
    Minimal-invasive pre-train audit:
    1) print every trainable tensor (name/shape/numel),
    2) summarize total/trainable ratio,
    3) aggregate by prefix + suspicious keywords,
    4) print PEFT config and heuristic diagnosis evidence.
    """
    if not _is_main_process():
        # Avoid duplicated huge logs under torchrun multi-process training.
        return

    print("\n========== [TrainableAudit] START ==========")
    totals = get_parameter_number(model)
    total_params = totals["Total"]
    trainable_params = totals["Trainable"]
    trainable_ratio = (100.0 * trainable_params / total_params) if total_params else 0.0
    print(
        f"[TrainableAudit] Parameter summary: total={total_params:,}, "
        f"trainable={trainable_params:,}, ratio={trainable_ratio:.4f}%"
    )

    trainable_items = []
    non_lora_items = []
    lora_param_numel = 0
    non_lora_param_numel = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        if numel == 0 and hasattr(param, "ds_numel"):
            numel = param.ds_numel
        shape = tuple(param.shape)
        trainable_items.append((name, shape, int(numel)))
        lowered = name.lower()
        if "lora_a" in lowered or "lora_b" in lowered:
            lora_param_numel += int(numel)
        else:
            non_lora_param_numel += int(numel)
            non_lora_items.append((name, shape, int(numel)))

    print(f"[TrainableAudit] Trainable tensor count: {len(trainable_items)}")
    print("[TrainableAudit] Full requires_grad=True parameter list (name | shape | numel):")
    for idx, (name, shape, numel) in enumerate(trainable_items):
        print(f"[TrainableAudit]   {idx:04d} | {name} | shape={shape} | numel={numel:,}")

    print(
        f"[TrainableAudit] LoRA vs non-LoRA split: "
        f"lora={lora_param_numel:,}, non_lora={non_lora_param_numel:,}"
    )

    for depth in (2, 3):
        grouped = _aggregate_trainable_by_prefix(trainable_items, depth=depth)
        print(f"[TrainableAudit] Trainable params aggregated by first {depth} module levels (top 40):")
        for prefix, count in grouped[:40]:
            print(f"[TrainableAudit]   {prefix}: {count:,}")

    # Keyword view for quick root-cause detection; overlapping matches are expected.
    tracked_keywords = [
        "lora_a",
        "lora_b",
        "vision",
        "visual",
        "resampler",
        "projector",
        "mlp",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    keyword_stats = {k: 0 for k in tracked_keywords}
    for name, _, numel in trainable_items:
        lowered = name.lower()
        for key in tracked_keywords:
            if key in lowered:
                keyword_stats[key] += numel

    print("[TrainableAudit] Trainable params by keyword (overlap allowed):")
    for key in tracked_keywords:
        print(f"[TrainableAudit]   {key}: {keyword_stats[key]:,}")

    peft_cfg = getattr(model, "peft_config", None)
    if peft_cfg is not None:
        print("[TrainableAudit] peft_config detected.")
        if isinstance(peft_cfg, dict):
            print(f"[TrainableAudit] adapter_names={list(peft_cfg.keys())}")
            for adapter_name, cfg in peft_cfg.items():
                print(f"[TrainableAudit] adapter={adapter_name}")
                print(f"[TrainableAudit]   r={getattr(cfg, 'r', None)}")
                print(f"[TrainableAudit]   lora_alpha={getattr(cfg, 'lora_alpha', None)}")
                print(f"[TrainableAudit]   lora_dropout={getattr(cfg, 'lora_dropout', None)}")
                print(f"[TrainableAudit]   target_modules={getattr(cfg, 'target_modules', None)}")
                print(f"[TrainableAudit]   modules_to_save={getattr(cfg, 'modules_to_save', None)}")
                print(f"[TrainableAudit]   layers_to_transform={getattr(cfg, 'layers_to_transform', None)}")
                print(f"[TrainableAudit]   bias={getattr(cfg, 'bias', None)}")
        else:
            print(f"[TrainableAudit] peft_config={peft_cfg}")
    elif lora_args is not None:
        print("[TrainableAudit] peft_config not found on model; fallback to lora_args:")
        print(f"[TrainableAudit]   lora_target_modules={getattr(lora_args, 'lora_target_modules', None)}")
        print(f"[TrainableAudit]   lora_r={getattr(lora_args, 'lora_r', None)}")
        print(f"[TrainableAudit]   lora_alpha={getattr(lora_args, 'lora_alpha', None)}")

    if hasattr(model, "print_trainable_parameters"):
        print("[TrainableAudit] model.print_trainable_parameters():")
        try:
            model.print_trainable_parameters()
        except Exception as exc:
            print(f"[TrainableAudit] print_trainable_parameters() failed: {exc}")

    # Heuristic diagnosis so logs explicitly state "why" trainable params are large.
    print("[TrainableAudit] Heuristic diagnosis:")
    non_lora_ratio = (100.0 * non_lora_param_numel / trainable_params) if trainable_params else 0.0
    if non_lora_param_numel > lora_param_numel:
        print(
            "[TrainableAudit]   Evidence: non-LoRA trainable params dominate "
            f"({non_lora_param_numel:,} / {trainable_params:,}, {non_lora_ratio:.2f}%)."
        )
        print(
            "[TrainableAudit]   Inference: trainable params are likely not only LoRA adapters; "
            "large base modules (e.g. embed_tokens/resampler/vision projector) are being optimized."
        )
        print("[TrainableAudit]   Top non-LoRA trainable tensors (top 20):")
        for name, shape, numel in sorted(non_lora_items, key=lambda x: x[2], reverse=True)[:20]:
            print(f"[TrainableAudit]     {name} | shape={shape} | numel={numel:,}")
    else:
        print("[TrainableAudit]   Evidence: LoRA params are the major trainable part.")
        print("[TrainableAudit]   Inference: if trainable count is still high, inspect LoRA target_modules scope.")

    if keyword_stats["embed_tokens"] > 0 or keyword_stats["resampler"] > 0:
        print(
            "[TrainableAudit]   Evidence: embed_tokens/resampler appear in trainable set. "
            "This is often the direct reason for >100M trainable params in LoRA runs."
        )
    if keyword_stats["vision"] > 0 or keyword_stats["visual"] > 0 or keyword_stats["projector"] > 0:
        print(
            "[TrainableAudit]   Evidence: vision/visual/projector trainable params are present. "
            "Multimodal modules are participating in optimization."
        )

    print("========== [TrainableAudit] END ==========\n")


def run_federated_finetune(model_args, data_args, training_args, lora_args, fed_args, hooks=None):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(training_args.cache_dir, exist_ok=True)
    os.makedirs(training_args.output_dir, exist_ok=True)
    hook_manager = FederatedHookManager(hooks)
    model, tokenizer = build_model_and_tokenizer(model_args, training_args, lora_args)
    global_dict = {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()}
    local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

    slice_config = _build_slice_config(model, training_args.max_slice_nums)
    _sync_vision_batch_size(
        model=model,
        max_slice_nums=training_args.max_slice_nums,
        configured_vbs=getattr(training_args, "vision_batch_size", None),
    )
    llm_type = training_args.llm_type
    patch_size = getattr(model.config, "patch_size", 14)
    query_num = getattr(model.config, "query_num", 64)
    batch_vision = getattr(model.config, "batch_vision_input", False)
    task_type = normalize_task_type(getattr(data_args, "task_type", "classification"))
    data_format = getattr(data_args, "data_format", "auto")
    train_split = getattr(data_args, "train_split", "train")
    task_loader_kwargs = build_task_loader_kwargs(task_type=task_type, data_args=data_args)

    local_datasets = []
    sample_num_list = []
    bad_sample_logs = []
    for i in range(fed_args.num_clients):
        client_data_path = resolve_client_data_path(data_args.data_path, i)
        bad_sample_log_path = os.path.join(training_args.output_dir, f"bad_samples_client_{i}.jsonl")
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_path=client_data_path,
            llm_type=llm_type,
            processor=getattr(model, "processor", None),
            model_name_or_path=model_args.model_name_or_path,
            slice_config=slice_config,
            patch_size=patch_size,
            query_nums=query_num,
            batch_vision=batch_vision,
            max_length=training_args.model_max_length,
            enable_audio=training_args.enable_audio,
            bad_sample_log_path=bad_sample_log_path,
            task_type=task_type,
            data_format=data_format,
            split=train_split,
            task_loader_kwargs=task_loader_kwargs,
        )
        local_datasets.append(data_module)
        sample_num_list.append(len(data_module["train_dataset"]))
        bad_sample_logs.append(bad_sample_log_path)
    _print_first_training_sample_snapshot(local_datasets, tokenizer)

    training_loss = [[] for _ in range(fed_args.num_clients)]
    training_metrics = [[] for _ in range(fed_args.num_clients)]
    fednova_stats_list = [None for _ in range(fed_args.num_clients)]
    # Keep deterministic reproducibility, while avoiding repeated identical
    # local mini-subsets when max_steps is set to a small positive value.
    base_seed = int(getattr(training_args, "seed", 42))
    base_data_seed = int(getattr(training_args, "data_seed", base_seed))
    if training_args.use_lora:
        _audit_trainable_parameters(model, lora_args=lora_args)
    else:
        print(get_parameter_number(model))
    hook_manager.emit("on_train_start", {
        "num_rounds": fed_args.num_rounds,
        "num_clients": fed_args.num_clients,
        "sample_clients": fed_args.sample_clients,
        "global_state_summary": summarize_state_dict(global_dict),
        "model": model,
        "tokenizer": tokenizer,
    })
    outer_lr_schedule = getattr(fed_args, "outer_lr_schedule", "cosine")
    outer_lr_eta_min = float(getattr(fed_args, "outer_lr_eta_min", 0.0))
    outer_lr_warmup_rounds = _resolve_outer_lr_warmup_rounds(
        fed_args.num_rounds,
        getattr(fed_args, "outer_lr_warmup_rounds", None),
    )
    if _is_main_process():
        print(
            "[OuterLR] "
            f"schedule={outer_lr_schedule} "
            f"base_lr={float(fed_args.init_learning_rate):.10g} "
            f"eta_min={outer_lr_eta_min:.10g} "
            f"warmup_rounds={outer_lr_warmup_rounds} "
            "inner_lr_schedule=constant",
            flush=True,
        )
    last_trainer = None
    for round_idx in tqdm(range(fed_args.num_rounds)):
        clients_this_round = get_clients_this_round(fed_args, round_idx)
        hook_manager.emit("on_round_start", {
            "round_idx": round_idx,
            "clients_this_round": clients_this_round,
        })
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)
                training_metrics[client].append({})
                continue
            # Ensure each (round, client) gets a distinct but reproducible
            # dataloader shuffle order.
            per_client_offset = round_idx * fed_args.num_clients + client
            training_args.seed = base_seed + per_client_offset
            training_args.data_seed = base_data_seed + per_client_offset
            set_peft_model_state_dict(model, global_dict)
            sub_dataset = local_datasets[client]
            training_args.learning_rate = compute_outer_learning_rate(
                round_idx=round_idx,
                base_lr=fed_args.init_learning_rate,
                total_rounds=fed_args.num_rounds,
                schedule=outer_lr_schedule,
                warmup_rounds=outer_lr_warmup_rounds,
                eta_min=outer_lr_eta_min,
            )
            # Keep outer round-level LR schedule, but disable inner per-step linear decay.
            training_args.lr_scheduler_type = "constant"
            training_args.warmup_steps = 0
            training_args.warmup_ratio = 0.0
            hook_manager.emit("on_client_start", {
                "round_idx": round_idx,
                "client_idx": client,
                "learning_rate": training_args.learning_rate,
            })
            if _alg_match(fed_args.fed_alg, "fedprox"):
                if "base" in fed_args.fed_alg:
                    s_layer = 0
                    mu_w = 1.0
                elif "adaptive" in fed_args.fed_alg:
                    mu_w = fed_args.mu_w
                    s_layer = fed_args.s_layer
                else:
                    mu_w = fed_args.mu_w
                    s_layer = fed_args.s_layer
                trainer = CPMTrainerReg(
                    global_state=global_dict,
                    prox_mu=fed_args.prox_mu,
                    mu_w=mu_w,
                    s_layer=s_layer,
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **sub_dataset,
                )
            elif _alg_match(fed_args.fed_alg, "scaffold"):
                trainer = CPMTrainerScaffold(
                    global_auxiliary=global_auxiliary,
                    local_auxiliary=auxiliary_model_list[client],
                    scaffold_lr=training_args.learning_rate,
                    scaffold_eps=fed_args.scaffold_eps,
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **sub_dataset,
                )
            elif _alg_match(fed_args.fed_alg, "fednova"):
                trainer = CPMTrainerFedNova(
                    fednova_eps=fed_args.fednova_eps,
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **sub_dataset,
                )
            else:
                trainer = CPMTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **sub_dataset,
                )
            trainer.fed_round_idx = round_idx
            trainer.fed_client_idx = client
            last_trainer = trainer
            results = trainer.train()
            training_loss[client].append(results.training_loss)
            local_metrics = trainer.get_federated_metrics()
            if _alg_match(fed_args.fed_alg, "scaffold"):
                matched = local_metrics.get("scaffold_matched_params")
                corr_norm = local_metrics.get("scaffold_corr_norm")
                if matched == 0:
                    msg = (
                        f"[SCAFFOLD][WARN] round={round_idx} client={client} "
                        "matched_params=0 (control variate is not being applied)."
                    )
                    if _env_flag("OPENFED_STRICT_SCAFFOLD", default=False):
                        raise RuntimeError(msg)
                    print(msg, flush=True)
                if not _is_finite_number(corr_norm):
                    raise RuntimeError(
                        f"[SCAFFOLD][ERROR] Non-finite correction norm: "
                        f"round={round_idx} client={client} corr_norm={corr_norm}"
                    )
            if _alg_match(fed_args.fed_alg, "fednova"):
                fednova_stats = trainer.get_fednova_stats(results)
                fednova_stats_list[client] = fednova_stats
                local_metrics.update(fednova_stats)
            training_metrics[client].append(local_metrics)
            local_dict_list[client] = {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()}
            if _alg_match(fed_args.fed_alg, "scaffold"):
                local_steps = getattr(results, "global_step", None)
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.update_auxiliary(
                    global_state_before=global_dict,
                    local_state_after=local_dict_list[client],
                    local_steps=local_steps,
                )
            hook_manager.emit("on_client_end", {
                "round_idx": round_idx,
                "client_idx": client,
                "training_loss": results.training_loss,
                "metrics": local_metrics,
                "fednova_stats": fednova_stats_list[client] if _alg_match(fed_args.fed_alg, "fednova") else None,
                "parameter_delta_l2": _state_delta_l2(local_dict_list[client], global_dict),
                "local_state_summary": summarize_state_dict(local_dict_list[client]),
            })

        global_dict, global_auxiliary = global_aggregate(
            fed_args,
            global_dict,
            local_dict_list,
            sample_num_list,
            clients_this_round,
            round_idx,
            proxy_dict=proxy_dict,
            opt_proxy_dict=opt_proxy_dict,
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
            fednova_info=(fednova_stats_list,),
        )
        set_peft_model_state_dict(model, global_dict)
        hook_manager.emit("on_aggregate_end", {
            "round_idx": round_idx,
            "global_state_summary": summarize_state_dict(global_dict),
            "global_auxiliary": global_auxiliary,
        })
        if (round_idx + 1) % fed_args.save_model_freq == 0:
            if last_trainer is not None:
                last_trainer.save_state()
                last_trainer.save_model(os.path.join(training_args.output_dir, f"checkpoint-{round_idx + 1}"))
        np.save(os.path.join(training_args.output_dir, "training_loss.npy"), np.array(training_loss))
        np.save(os.path.join(training_args.output_dir, "training_metrics.npy"), np.array(training_metrics, dtype=object))
        hook_manager.emit("on_round_end", {
            "round_idx": round_idx,
            "clients_this_round": clients_this_round,
        })
    hook_manager.emit("on_train_end", {
        "training_loss": training_loss,
        "training_metrics": training_metrics,
        "global_state_summary": summarize_state_dict(global_dict),
        "bad_sample_logs": bad_sample_logs,
    })


def run_federated_finetune_from_config(exp_args):
    model_args = ModelArguments(**dict(exp_args.model_args))
    data_args = DataArguments(**dict(exp_args.data_args))
    training_args = TrainingArguments(**dict(exp_args.training_args))
    lora_args = LoraArguments(**dict(exp_args.lora_args))
    fed_args = FedArguments(**dict(exp_args.fed_args))
    hooks = getattr(exp_args, "hooks", None)
    run_federated_finetune(model_args, data_args, training_args, lora_args, fed_args, hooks=hooks)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments, FedArguments))
    model_args, data_args, training_args, lora_args, fed_args = parser.parse_args_into_dataclasses()
    run_federated_finetune(model_args, data_args, training_args, lora_args, fed_args)


if __name__ == "__main__":
    main()
