from easydict import EasyDict


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
        data_path="./data/crisis-mmd/minicpmv_data/partition-alpha0.5-clt10",
        eval_data_path="./data/crisis-mmd/minicpmv_data/test.json",
    ),
    training_args=dict(
        output_dir="./mllmzoo/output/qwen2vl_crisismmid_fedavg_quick",
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
        fed_alg="fedavg",
        num_rounds=6,
        num_clients=10,
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
        eval_data_path="./data/crisis-mmd/minicpmv_data/test.json",
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

    run_federated_with_metrics_hook(exp_args, run_name="qwen2vl_crisismmid_fedavg_quick")
