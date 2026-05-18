import os
from pathlib import Path


def _get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_hateful_runtime():
    root = Path(os.environ.get("OPENFED_HATEFUL_MEMES_ROOT", "./hateful_memes"))
    stat_setting = os.environ.get("OPENFED_HATEFUL_MEMES_STAT", "dir_0.1")
    modal_setting = os.environ.get("OPENFED_HATEFUL_MEMES_MODAL", "aligned")
    explicit_data_path = os.environ.get("OPENFED_HATEFUL_MEMES_DATA_PATH")

    if explicit_data_path:
        data_path = Path(explicit_data_path)
    else:
        data_path = root / "federated" / stat_setting / modal_setting

    eval_path = Path(os.environ.get("OPENFED_HATEFUL_MEMES_EVAL_PATH", str(root / "dev.jsonl")))
    output_dir = os.environ.get(
        "OPENFED_HATEFUL_MEMES_OUTPUT_DIR",
        f"./outputs/checkpoints/minicpmv_hateful_memes_{stat_setting}_{modal_setting}_quick",
    )
    return {
        "root": str(root),
        "data_path": str(data_path),
        "eval_path": str(eval_path),
        "output_dir": output_dir,
        "stat_setting": stat_setting,
        "modal_setting": modal_setting,
    }


_runtime = _resolve_hateful_runtime()


exp_args = dict(
    model_args=dict(
        model_name_or_path=os.environ.get("OPENFED_MINICPM_MODEL", "openbmb/MiniCPM-V-2_6-int4"),
    ),
    data_args=dict(
        data_path=_runtime["data_path"],
        eval_data_path=_runtime["eval_path"],
        task_type="hateful_memes",
        data_format="jsonl",
        hateful_memes_root=_runtime["root"],
        stat_setting=_runtime["stat_setting"],
        modal_setting=_runtime["modal_setting"],
        strict_image_path=_get_env_bool("OPENFED_HATEFUL_MEMES_STRICT_IMAGE_PATH", False),
    ),
    training_args=dict(
        output_dir=_runtime["output_dir"],
        cache_dir=os.environ.get("OPENFED_CACHE_DIR", "./mllmzoo/cache"),
        seed=42,
        data_seed=42,
        full_determinism=True,
        num_train_epochs=_get_env_int("OPENFED_LOCAL_EPOCHS", 1),
        per_device_train_batch_size=_get_env_int("OPENFED_BATCH_SIZE", 1),
        gradient_accumulation_steps=_get_env_int("OPENFED_GRAD_ACCUM_STEPS", 1),
        learning_rate=_get_env_float("OPENFED_LR", 3e-5),
        bf16=_get_env_bool("OPENFED_BF16", True),
        fp16=_get_env_bool("OPENFED_FP16", False),
        logging_steps=_get_env_int("OPENFED_LOGGING_STEPS", 1),
        save_steps=_get_env_int("OPENFED_SAVE_STEPS", 50),
        save_total_limit=_get_env_int("OPENFED_SAVE_TOTAL_LIMIT", 1),
        remove_unused_columns=False,
        report_to="none",
        model_max_length=_get_env_int("OPENFED_MODEL_MAX_LENGTH", 512),
        max_slice_nums=_get_env_int("OPENFED_MAX_SLICE_NUMS", 4),
        vision_batch_size=_get_env_int("OPENFED_VISION_BATCH_SIZE", 8),
        llm_type="minicpm",
        gradient_checkpointing=True,
        tune_vision=False,
        tune_llm=False,
        use_lora=True,
        enable_audio=False,
        max_steps=_get_env_int("OPENFED_MAX_STEPS", 1),
    ),
    lora_args=dict(
        lora_r=_get_env_int("OPENFED_LORA_R", 4),
        lora_alpha=_get_env_int("OPENFED_LORA_ALPHA", 8),
        lora_dropout=_get_env_float("OPENFED_LORA_DROPOUT", 0.05),
        lora_bias="none",
        q_lora=False,
    ),
    fed_args=dict(
        fed_alg=os.environ.get("OPENFED_FED_ALG", "fedavg"),
        num_rounds=_get_env_int("OPENFED_NUM_ROUNDS", 1),
        num_clients=_get_env_int("OPENFED_NUM_CLIENTS", 10),
        sample_clients=_get_env_int("OPENFED_SAMPLE_CLIENTS", 2),
        init_learning_rate=_get_env_float("OPENFED_OUTER_LR", 3e-5),
        outer_lr_schedule="cosine",
        outer_lr_eta_min=_get_env_float("OPENFED_OUTER_LR_ETA_MIN", 5e-6),
        save_model_freq=_get_env_int("OPENFED_SAVE_MODEL_FREQ", 1),
        prox_mu=0.01,
        mu_w=0.1,
        s_layer=4,
        fedopt_tau=1e-3,
        fedopt_eta=1e-3,
        fedopt_beta1=0.9,
        fedopt_beta2=0.99,
    ),
    eval_args=dict(
        eval_data_path=_runtime["eval_path"],
        task_type="hateful_memes",
        data_format="jsonl",
        hateful_memes_root=_runtime["root"],
        strict_image_path=_get_env_bool("OPENFED_HATEFUL_MEMES_STRICT_IMAGE_PATH", False),
        eval_freq=_get_env_int("OPENFED_EVAL_FREQ", 1),
        max_new_tokens=_get_env_int("OPENFED_EVAL_MAX_NEW_TOKENS", 4),
        max_samples=_get_env_int("OPENFED_EVAL_MAX_SAMPLES", 32),
        startup_eval_full=_get_env_bool("OPENFED_STARTUP_EVAL_FULL", False),
        final_eval_full=_get_env_bool("OPENFED_FINAL_EVAL_FULL", True),
    ),
    run_args=dict(
        mode=os.environ.get("OPENFED_RUN_MODE", "federated"),
        enable_final_eval=_get_env_bool("OPENFED_ENABLE_FINAL_EVAL", True),
    ),
)


if __name__ == "__main__":
    from fling_mllm.pipeline import run_mode_from_config

    if not os.path.isdir(exp_args["data_args"]["data_path"]):
        raise FileNotFoundError(
            "Hateful Memes federated data_path does not exist: "
            f"{exp_args['data_args']['data_path']!r}. Check OPENFED_HATEFUL_MEMES_STAT and "
            "OPENFED_HATEFUL_MEMES_MODAL."
        )
    run_mode_from_config(exp_args)
