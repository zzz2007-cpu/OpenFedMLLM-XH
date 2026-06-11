import os
from pathlib import Path

from easydict import EasyDict


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def build_scienceqa_chi_exp(
    algorithm: str,
    setting: str,
    run_name: str,
    ablation: str | None = None,
):
    fed_root = Path(os.environ.get("SCIENCEQA_FED_DIR", "./data/scienceqa/federated_chi"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", "./outputs/scienceqa_chi"))
    seed = _env_int("SEED", 42)
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    semantic_split_target = os.environ.get("SCIENCEQA_SPLIT_TARGET", "subject")
    output_dir = output_root / run_name / f"seed_{seed}"

    enable_shared = ablation != "no_shared_lora"
    enable_modality = ablation not in {"no_modality_lora", "shared_lora_only"}
    enable_image = enable_modality
    enable_text = enable_modality
    lambda_cons = 0.0 if ablation == "no_consistency" else _env_float("LAMBDA_CONS", 0.1)
    lambda_label = 0.0 if ablation == "no_hetagg" else _env_float("LAMBDA_LABEL", 1.0)
    lambda_modality = 0.0 if ablation == "no_hetagg" else _env_float("LAMBDA_MODALITY", 1.0)
    if ablation == "shared_lora_only":
        enable_shared = True

    return EasyDict(
        model_args=dict(
            model_name_or_path=model_path,
            model_family="qwen2_vl",
            trust_remote_code=True,
            attn_implementation=os.environ.get("ATTN_IMPLEMENTATION") or None,
            processor_min_pixels=_env_int("PROCESSOR_MIN_PIXELS", 256 * 28 * 28),
            processor_max_pixels=_env_int("PROCESSOR_MAX_PIXELS", 1024 * 28 * 28),
        ),
        data_args=dict(
            data_path=str(fed_root / setting),
            eval_data_path=str(fed_root / "validation.jsonl"),
            task_type="scienceqa",
            data_format="jsonl",
            train_split="train",
            eval_split="eval",
            strict_image_path=True,
            max_train_samples_per_client=(
                _env_int("MAX_TRAIN_SAMPLES_PER_CLIENT", 0) or None
            ),
        ),
        training_args=dict(
            output_dir=str(output_dir),
            cache_dir=os.environ.get("CACHE_DIR", "./mllmzoo/cache"),
            seed=seed,
            data_seed=seed,
            full_determinism=True,
            num_train_epochs=_env_int("LOCAL_EPOCHS", 1),
            per_device_train_batch_size=_env_int("BATCH_SIZE", 1),
            gradient_accumulation_steps=_env_int("GRAD_ACCUM", 8),
            learning_rate=_env_float("LR", 5e-5),
            bf16=os.environ.get("BF16", "1") == "1",
            fp16=os.environ.get("FP16", "0") == "1",
            logging_steps=_env_int("LOGGING_STEPS", 20),
            save_steps=_env_int("SAVE_STEPS", 1000),
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",
            model_max_length=_env_int("MODEL_MAX_LENGTH", 1024),
            llm_type="qwen2_vl",
            gradient_checkpointing=os.environ.get("GRADIENT_CHECKPOINTING", "1") == "1",
            tune_vision=False,
            tune_llm=False,
            use_lora=True,
            enable_audio=False,
            max_steps=_env_int("MAX_STEPS", -1),
        ),
        lora_args=dict(
            lora_r=_env_int("LORA_R", 8),
            lora_alpha=_env_int("LORA_ALPHA", 16),
            lora_dropout=_env_float("LORA_DROPOUT", 0.05),
            lora_target_modules=os.environ.get("LORA_TARGET_MODULES", "auto"),
            fedchi_decomposed_lora=algorithm.lower().startswith("fedchi"),
            fedchi_shared_target_modules=os.environ.get("FEDCHI_SHARED_TARGET_MODULES", "auto"),
            fedchi_image_target_modules=os.environ.get("FEDCHI_IMAGE_TARGET_MODULES", "auto"),
            fedchi_text_target_modules=os.environ.get("FEDCHI_TEXT_TARGET_MODULES", "auto"),
            fedchi_enable_shared_adapter=enable_shared,
            fedchi_enable_image_adapter=enable_image,
            fedchi_enable_text_adapter=enable_text,
            lora_bias="none",
            q_lora=os.environ.get("Q_LORA", "0") == "1",
        ),
        fed_args=dict(
            fed_alg=algorithm,
            num_rounds=_env_int("ROUNDS", 20),
            num_clients=_env_int("NUM_CLIENTS", 10),
            sample_clients=_env_int("CLIENTS_PER_ROUND", 5),
            init_learning_rate=_env_float("OUTER_LR", 5e-5),
            save_model_freq=_env_int("SAVE_MODEL_FREQ", 5),
            prox_mu=_env_float("MU_PROX", 0.01),
            mu_w=0.1,
            s_layer=4,
            lambda_label=lambda_label,
            lambda_modality=lambda_modality,
            lambda_cons=lambda_cons,
            fedchi_weights_path=str(output_dir / "fedchi_weights.jsonl"),
            fedopt_tau=1e-3,
            fedopt_eta=1e-3,
            fedopt_beta1=0.9,
            fedopt_beta2=0.99,
        ),
        eval_args=dict(
            eval_data_path=str(fed_root / "validation.jsonl"),
            task_type="scienceqa",
            data_format="jsonl",
            eval_split="eval",
            eval_freq=_env_int("EVAL_EVERY", 1),
            max_new_tokens=_env_int("EVAL_MAX_NEW_TOKENS", 8),
            max_samples=_env_int("EVAL_MAX_SAMPLES", 500),
            startup_eval_full=False,
            final_eval_full=True,
            dataset="ScienceQA/image_present",
            model=model_path,
            algorithm=algorithm,
            setting=setting,
            label_level=setting.split("_")[0],
            semantic_level=setting.split("_")[0],
            heterogeneity_axis="semantic",
            semantic_split_target=semantic_split_target,
            modality_level=setting.split("_")[1],
            ablation=ablation,
            seed=seed,
        ),
        run_args=dict(
            mode="federated",
            append_mode_subdir=False,
            enable_final_eval=True,
        ),
        chi_metadata=dict(
            dataset="ScienceQA/image_present",
            model=model_path,
            algorithm=algorithm,
            setting=setting,
            heterogeneity_axis="semantic",
            semantic_split_target=semantic_split_target,
            ablation=ablation,
            seed=seed,
        ),
    )
