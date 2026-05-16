from easydict import EasyDict


exp_args = dict(
    model_args=dict(
        model_name_or_path="openbmb/MiniCPM-V-2_6-int4",
    ),
    data_args=dict(
        data_path="./data/crisis-mmd/minicpmv_data/partition-alpha0.5-clt10",
        eval_data_path="./data/crisis-mmd/minicpmv_data/test.json",
    ),
    training_args=dict(
        output_dir="./mllmzoo/output/tiny_smoke",
        cache_dir="./mllmzoo/cache",
        num_train_epochs=1,
        max_steps=50,                      # slightly stronger than baseline (2 -> 10)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        bf16=True,
        fp16=False,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",
        model_max_length=512,             # 短序列，省显存
        max_slice_nums=9,                 # 使用统一切片配置
        vision_batch_size=13,
        llm_type="minicpm",
        gradient_checkpointing=False,     # smoke时关掉，省时间
        tune_vision=False,
        tune_llm=False,
        use_lora=True,
        enable_audio=False,
    ),
    lora_args=dict(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
        q_lora=True,
    ),
    fed_args=dict(
        fed_alg="fedavg",
        num_rounds=40,                     # slightly longer tiny run (2 -> 3)
        num_clients=6,                    # 只用6个客户端
        sample_clients=4,
        init_learning_rate=1e-4,
        save_model_freq=100,
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
        eval_freq=1,                      # 每轮都测，验证eval不崩
        max_new_tokens=16,                # 短输出，省时间
        max_samples=100,                   # 只测100条样本
    ),
)

exp_args = EasyDict(exp_args)


if __name__ == "__main__":
    import torch
    from fling_mllm.pipeline import run_federated_finetune_from_config
    from fling_mllm.federated.hooks import FederatedHook

    class PrintMetricsHook(FederatedHook):
        """每轮打印 loss、GPU显存、Acc/F1，无复杂依赖。"""
        def __init__(self, eval_args, output_dir):
            self._ea = eval_args
            self._output_dir = output_dir
            self._round_losses = {}
            self._clients_this_round = {}  # {round_idx: [client_ids]}
            self._total_comm_bytes = 0    # 累计通信量（字节）
            self._model = None
            self._tokenizer = None

            import os
            ep = getattr(eval_args, "eval_data_path", None)
            if ep and os.path.exists(ep):
                self._eval_data_path = ep
                self._eval_ready = True
            else:
                self._eval_data_path = None
                self._eval_ready = False
                print("[PrintMetricsHook] 未找到 eval_data_path，跳过每轮评估。")

        def on_train_start(self, context):
            self._model = context.get("model")
            self._tokenizer = context.get("tokenizer")

        def on_round_start(self, context):
            r = context.get("round_idx", -1)
            self._clients_this_round[r] = context.get("clients_this_round", [])

        def on_client_end(self, context):
            r = context.get("round_idx", -1)
            loss = context.get("training_loss", float("nan"))
            client = context.get("client_idx", -1)
            self._round_losses.setdefault(r, []).append((client, loss))

        def on_aggregate_end(self, context):
            r = context.get("round_idx", -1)
            losses = self._round_losses.get(r, [])

            loss_str = "  ".join(f"client{c}={l:.4f}" for c, l in sorted(losses))
            print(f"\n{'='*60}")
            print(f"[Round {r+1}] Loss → {loss_str if loss_str else 'N/A'}")

            if torch.cuda.is_available():
                alloc  = torch.cuda.memory_allocated() / 1024**3
                reserv = torch.cuda.memory_reserved()  / 1024**3
                print(f"[Round {r+1}] GPU  → allocated={alloc:.2f}GB  reserved={reserv:.2f}GB")

            # ── 通信量统计 ────────────────────────────────────────────────
            state_summary = context.get("global_state_summary", {})
            param_count = state_summary.get("param_count", 0)
            clients_this_round = self._clients_this_round.get(r, [])
            n_clients = len(clients_this_round) if clients_this_round else 1
            BYTES_PER_PARAM = 2  # bfloat16
            round_comm_bytes = 2 * n_clients * param_count * BYTES_PER_PARAM
            self._total_comm_bytes += round_comm_bytes
            def _fmt(b):
                if b >= 1024**3: return f"{b/1024**3:.3f} GB"
                if b >= 1024**2: return f"{b/1024**2:.3f} MB"
                return f"{b/1024:.3f} KB"
            print(
                f"[Round {r+1}] Comm → "
                f"本轮={_fmt(round_comm_bytes)}  "
                f"累计={_fmt(self._total_comm_bytes)}  "
                f"(LoRA参数 {param_count:,}  ×  {n_clients}客户端)"
            )

            if self._eval_ready and self._model is not None:
                freq = getattr(self._ea, "eval_freq", 1)
                if (r + 1) % freq == 0:
                    self._run_eval(r)
            print(f"{'='*60}\n")

        def _run_eval(self, round_idx):
            import json, os, random
            from fling_mllm.utils.eval_utils import (
                correct_image_paths, extract_question, extract_ground_truth,
                extract_prediction_letter, generate_answer,
                generate_label_probability_scores, match_prediction_to_label,
                compute_classification_metrics,
            )
            with open(self._eval_data_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            test_data = correct_image_paths(test_data, os.getcwd())
            max_s = getattr(self._ea, "max_samples", None)
            if max_s is not None and max_s > 0 and len(test_data) > max_s:
                sample_seed = int(getattr(self._ea, "eval_sample_seed", 42))
                rng = random.Random(sample_seed)
                test_data = rng.sample(test_data, max_s)
            label_set = sorted(set(extract_ground_truth(s) for s in test_data))
            print(f"[DEBUG] label_set ({len(label_set)} classes): {label_set}")

            self._model.eval()
            preds, gts, score_matrix = [], [], []
            import torch as _torch
            for i, sample in enumerate(test_data):
                q, gt, img = extract_question(sample), extract_ground_truth(sample), sample.get("image")
                try:
                    with _torch.no_grad():
                        raw = generate_answer(
                            model=self._model, tokenizer=self._tokenizer,
                            question=q, image_path=img,
                            max_new_tokens=getattr(self._ea, "max_new_tokens", 16),
                        )
                except Exception as e:
                    raw = ""
                    print(f"[DEBUG] sample {i} 推理异常: {e}")
                pred_letter = extract_prediction_letter(raw, label_set=label_set, question=q)
                matched = match_prediction_to_label(raw, label_set, question=q)
                # 打印全部样本的原始输出（tiny只有10条，所以全打印）
                correct_mark = "✓" if matched == gt else "✗"
                print(
                    f"[DEBUG #{i}] {correct_mark}  raw={raw!r}  "
                    f"letter={pred_letter!r}  matched={matched!r}  gt={gt!r}"
                )
                preds.append(matched)
                score_matrix.append(
                    generate_label_probability_scores(
                        model=self._model,
                        tokenizer=self._tokenizer,
                        question=q,
                        label_set=label_set,
                        image_path=img,
                        max_new_tokens=getattr(self._ea, "score_max_new_tokens", 192),
                        fallback_text=raw,
                    )
                )
                gts.append(gt)
            self._model.train()

            m = compute_classification_metrics(preds, gts, label_set, score_matrix=score_matrix)
            auc_macro = m.get("auc_ovr_macro")
            auc_str = f"{auc_macro:.4f}" if auc_macro is not None else "N/A"
            auc_note = m.get("auc_note", "auc_note_unset")
            auc_rows = f"{m.get('auc_valid_rows', 0)}/{m.get('auc_total_rows', 0)}"
            auc_cls = m.get("auc_present_classes", 0)
            auc_skip = m.get("auc_skipped_rows", {})
            print(
                f"[Round {round_idx+1}] Eval → "
                f"Acc={m['accuracy']:.4f}  F1-W={m['f1_weighted']:.4f}  "
                f"F1-M={m['f1_macro']:.4f}  AUC-M={auc_str}  "
                f"(n={m['num_samples']}, auc_rows={auc_rows}, auc_cls={auc_cls}, auc_note={auc_note})"
            )
            print(
                f"[Round {round_idx+1}] AUC-Diag → "
                f"present_labels={m.get('auc_present_labels', [])}  "
                f"skipped={auc_skip}"
            )
            os.makedirs(self._output_dir, exist_ok=True)
            rec = {"round_idx": round_idx, **{k: m[k] for k in
                   ("accuracy", "f1_weighted", "f1_macro", "auc_ovr_macro", "auc_ovr_weighted",
                    "auc_note", "auc_valid_rows", "auc_total_rows", "auc_present_classes",
                    "auc_present_labels", "auc_skipped_rows", "num_samples")}}
            with open(os.path.join(self._output_dir, "eval_metrics_per_round.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    hooks = [PrintMetricsHook(exp_args.eval_args, exp_args.training_args.output_dir)]
    exp_args.hooks = hooks
    run_federated_finetune_from_config(exp_args)
