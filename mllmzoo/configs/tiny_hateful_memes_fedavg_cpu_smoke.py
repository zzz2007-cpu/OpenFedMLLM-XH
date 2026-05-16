import os
from pathlib import Path

from easydict import EasyDict


def _get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


root = Path(os.environ.get("OPENFED_HATEFUL_MEMES_ROOT", "./hateful_memes"))
stat_setting = os.environ.get("OPENFED_HATEFUL_MEMES_STAT", "iid")
modal_setting = os.environ.get("OPENFED_HATEFUL_MEMES_MODAL", "aligned")
data_path = Path(os.environ.get(
    "OPENFED_HATEFUL_MEMES_DATA_PATH",
    str(root / "federated" / stat_setting / modal_setting),
))
eval_path = Path(os.environ.get("OPENFED_HATEFUL_MEMES_EVAL_PATH", str(root / "dev.jsonl")))
output_dir = os.environ.get(
    "OPENFED_HATEFUL_MEMES_OUTPUT_DIR",
    "./outputs/checkpoints/hm_tiny_cpu_fedavg_smoke",
)


exp_args = EasyDict(
    model_args=dict(
        model_name_or_path="tiny_hateful_memes_cpu",
        trust_remote_code=False,
    ),
    data_args=dict(
        data_path=str(data_path),
        eval_data_path=str(eval_path),
        task_type="hateful_memes",
        data_format="jsonl",
        train_split="train",
        eval_split="eval",
        hateful_memes_root=str(root),
        stat_setting=stat_setting,
        modal_setting=modal_setting,
        strict_image_path=True,
        max_train_samples_per_client=_get_env_int("OPENFED_MAX_TRAIN_SAMPLES_PER_CLIENT", 8),
    ),
    training_args=dict(
        output_dir=output_dir,
        cache_dir="./mllmzoo/cache",
        seed=42,
        data_seed=42,
        num_train_epochs=_get_env_int("OPENFED_LOCAL_EPOCHS", 1),
        per_device_train_batch_size=_get_env_int("OPENFED_BATCH_SIZE", 4),
        gradient_accumulation_steps=1,
        learning_rate=_get_env_float("OPENFED_LR", 0.2),
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        max_steps=_get_env_int("OPENFED_MAX_STEPS", 2),
    ),
    lora_args=dict(),
    fed_args=dict(
        fed_alg="fedavg",
        num_rounds=_get_env_int("OPENFED_NUM_ROUNDS", 1),
        num_clients=_get_env_int("OPENFED_NUM_CLIENTS", 2),
        sample_clients=_get_env_int("OPENFED_SAMPLE_CLIENTS", 2),
        init_learning_rate=_get_env_float("OPENFED_OUTER_LR", 0.2),
        save_model_freq=1,
    ),
    eval_args=dict(
        eval_data_path=str(eval_path),
        task_type="hateful_memes",
        data_format="jsonl",
        hateful_memes_root=str(root),
        strict_image_path=True,
        max_samples=_get_env_int("OPENFED_EVAL_MAX_SAMPLES", 20),
    ),
    run_args=dict(
        mode="federated",
        enable_final_eval=False,
    ),
)
