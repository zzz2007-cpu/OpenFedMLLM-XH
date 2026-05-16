from easydict import EasyDict


exp_args = dict(
    model_args=dict(
        model_name_or_path="openbmb/MiniCPM-V-2_6-int4",
    ),
    data_args=dict(
        data_path="./partition_vqav2_supercat_dirichlet_clients10/alpha_1.0",
        eval_data_path="./VQAv2",
        task_type="vqa",
        data_format="auto",
        train_split="train",
        eval_split="val",
        vqa_image_root="./VQAv2",
        strict_image_path=True,
    ),
    training_args=dict(
        output_dir="./mllmzoo/output/minicpmv_vqav2_fedavg",
        cache_dir="./mllmzoo/cache",
        seed=42,
        data_seed=42,
        full_determinism=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=3e-5,
        bf16=True,
        fp16=False,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        model_max_length=1024,
        max_slice_nums=9,
        vision_batch_size=13,
        llm_type="minicpm",
        gradient_checkpointing=True,
        tune_vision=False,
        tune_llm=False,
        use_lora=True,
        enable_audio=False,
        max_steps=-1,
    ),
    lora_args=dict(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=r"llm\..*layers\.\d+\.self_attn\.(q_proj|v_proj)",
        lora_bias="none",
        q_lora=False,
    ),
    fed_args=dict(
        fed_alg="fedavg",
        num_rounds=20,
        num_clients=10,
        sample_clients=5,
        init_learning_rate=3e-5,
        outer_lr_schedule="cosine",
        outer_lr_eta_min=5e-6,
        save_model_freq=5,
        prox_mu=0.01,
        mu_w=0.1,
        s_layer=4,
        fedopt_tau=1e-3,
        fedopt_eta=1e-3,
        fedopt_beta1=0.9,
        fedopt_beta2=0.99,
    ),
    eval_args=dict(
        eval_data_path="./VQAv2",
        task_type="vqa",
        data_format="auto",
        eval_split="val",
        vqa_image_root="./VQAv2",
        eval_freq=1,
        max_new_tokens=16,
        max_samples=500,
        eval_sample_seed=42,
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
    from mllmzoo.configs._task_federated_monitor import run_federated_with_task_metrics_hook

    run_federated_with_task_metrics_hook(exp_args, run_name="minicpmv_vqav2_fedavg")
