import os
from pathlib import Path

from easydict import EasyDict


def _resolve_runtime_overrides(
    *,
    default_data_path,
    default_eval_path,
    default_output_dir,
    default_fed_alg,
    default_num_rounds,
    default_num_clients,
):
    explicit_data_path = os.environ.get("OPENFED_CRISISMMID_DATA_PATH")
    alpha = os.environ.get("OPENFED_CRISISMMID_ALPHA")
    data_root = os.environ.get("OPENFED_CRISISMMID_DATA_ROOT")
    num_clients = int(os.environ.get("OPENFED_CRISISMMID_NUM_CLIENTS", str(default_num_clients)))

    if explicit_data_path:
        data_path = explicit_data_path
    elif alpha:
        if not data_root:
            data_root = str(Path(default_data_path).parent)
        data_path = os.path.join(data_root, f"partition-alpha{alpha}-clt{num_clients}")
    else:
        data_path = default_data_path

    return {
        "data_path": data_path,
        "eval_data_path": os.environ.get("OPENFED_CRISISMMID_EVAL_PATH", default_eval_path),
        "output_dir": os.environ.get("OPENFED_CRISISMMID_OUTPUT_DIR", default_output_dir),
        "fed_alg": os.environ.get("OPENFED_CRISISMMID_FED_ALG", default_fed_alg),
        "num_rounds": int(os.environ.get("OPENFED_CRISISMMID_NUM_ROUNDS", str(default_num_rounds))),
        "num_clients": num_clients,
    }


_runtime = _resolve_runtime_overrides(
    default_data_path="./data/crisis-mmd/minicpmv_data/partition-alpha0.5-clt10",
    default_eval_path="./data/crisis-mmd/minicpmv_data/test.json",
    default_output_dir="./mllmzoo/output/qwen2vl_crisismmid_fedprox_quick",
    default_fed_alg="fedprox",
    default_num_rounds=6,
    default_num_clients=10,
)


exp_args = dict(
    model_args=dict(
        model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
        model_family="qwen2_vl",
        trust_remote_code=True,
        attn_implementation=None,
        processor_min_pixels=256 * 28 * 28,
        processor_max_pixels=1024 * 28 * 28,
    ),
    data_args=dict(
        data_path=_runtime["data_path"],
        eval_data_path=_runtime["eval_data_path"],
    ),
    training_args=dict(
        output_dir=_runtime["output_dir"],
        cache_dir="./mllmzoo/cache",
        seed=42,
        data_seed=42,
        full_determinism=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_steps=1000,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        model_max_length=1024,
        llm_type="qwen2_vl",
        gradient_checkpointing=True,
        tune_vision=False,
        tune_llm=False,
        use_lora=True,
        enable_audio=False,
        max_steps=20,
    ),
    lora_args=dict(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="auto",
        lora_bias="none",
        q_lora=False,
    ),
    fed_args=dict(
        fed_alg=_runtime["fed_alg"],
        num_rounds=_runtime["num_rounds"],
        num_clients=_runtime["num_clients"],
        sample_clients=4,
        init_learning_rate=5e-5,
        save_model_freq=2,
        prox_mu=0.01,
        mu_w=0.1,
        s_layer=4,
        fedopt_tau=1e-3,
        fedopt_eta=1e-3,
        fedopt_beta1=0.9,
        fedopt_beta2=0.99,
    ),
    eval_args=dict(
        eval_data_path=_runtime["eval_data_path"],
        eval_freq=1,
        max_new_tokens=16,
        max_samples=200,
        startup_eval_full=False,
        final_eval_full=True,
    ),
    run_args=dict(
        mode="federated",
        append_mode_subdir=False,
    ),
)

exp_args = EasyDict(exp_args)


if __name__ == "__main__":
    from mllmzoo.configs._federated_monitor import run_federated_with_metrics_hook

    run_federated_with_metrics_hook(exp_args, run_name="qwen2vl_crisismmid_fedprox_quick")
