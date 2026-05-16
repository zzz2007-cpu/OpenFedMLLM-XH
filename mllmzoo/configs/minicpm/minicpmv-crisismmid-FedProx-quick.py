import os
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
        output_dir="./mllmzoo/output/minicpmv_crisismmid_fedprox_quick_meaningful",
        cache_dir="./mllmzoo/cache",
        seed=42,
        data_seed=42,
        full_determinism=True,

        # 姣忚疆鏈湴璁粌寮哄害閫備腑
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,

        bf16=True,
        fp16=False,
        logging_steps=5,
        save_steps=1000,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to="none",

        # quick 妯″紡娌″繀瑕佹媺鍒?2048
        model_max_length=1024,
        max_slice_nums=9,
        vision_batch_size=13,
        llm_type="minicpm",
        gradient_checkpointing=True,
        tune_vision=False,
        tune_llm=False,
        use_lora=True,
        enable_audio=False,

        # 姣忎釜 client 姣忚疆鏈€澶?20 step锛屽鐪嬭秼鍔?        max_steps=int(os.environ.get("OPENFED_MAX_STEPS", "20")),
    ),
    lora_args=dict(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # Strict pure-LoRA: only attention q_proj / v_proj adapters.
        lora_target_modules=r"llm\..*layers\.\d+\.self_attn\.(q_proj|v_proj)",
        lora_bias="none",
        q_lora=False,
    ),
    fed_args=dict(
        fed_alg="fedprox",

        # 鐪熸 quick锛氬皯杞
        num_rounds=8,

        num_clients=10,

        # 姣忚疆鍙噰鏍烽儴鍒?client锛屾棦蹇張鏇撮€傚悎瑙傚療娉㈠姩
        sample_clients=4,

        init_learning_rate=3e-5,
        outer_lr_schedule="cosine",
        outer_lr_eta_min=5e-6,
        save_model_freq=4,
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

        # 姣忚疆閮界湅涓€娆?        eval_freq=1,

        # 鍒嗙被浠诲姟娌″繀瑕佺敓鎴愬お闀?        max_new_tokens=8,

        # 100 涓牱鏈冻澶熺湅瓒嬪娍
        max_samples=500,

        startup_eval_full=False,
        final_eval_full=False,
    ),
)

exp_args = EasyDict(exp_args)


if __name__ == "__main__":
    import random
    import sys
    import time
    import traceback
    import numpy as np
    import torch
    from fling_mllm.pipeline import run_federated_finetune_from_config
    from fling_mllm.federated.hooks import FederatedHook

    def _maybe_redirect_rank_logs(output_dir):
        # Auto-split per-rank logs for torchrun multi-process runs.
        enabled = os.environ.get("OPENFED_SPLIT_RANK_LOGS", "0") == "1"
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not enabled or world_size <= 1:
            return
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        run_id = os.environ.get("TORCHELASTIC_RUN_ID")
        if not run_id:
            run_id = f"port_{os.environ.get('MASTER_PORT', 'unknown')}"
        restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
        log_dir = os.path.join(output_dir, "rank_logs", f"{run_id}_r{restart}")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"rank_{rank}_local_{local_rank}.log")
        tee = os.environ.get("OPENFED_SPLIT_RANK_LOGS_TEE", "1") == "1"

        log_fp = open(log_path, "a", buffering=1, encoding="utf-8")
        if tee:
            class _Tee:
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, data):
                    for s in self.streams:
                        s.write(data)
                    return len(data)
                def flush(self):
                    for s in self.streams:
                        s.flush()
            sys.stdout = _Tee(sys.__stdout__, log_fp)
            sys.stderr = _Tee(sys.__stderr__, log_fp)
        else:
            sys.stdout = log_fp
            sys.stderr = log_fp
        print(
            f"[minicpmv_crisismmid_fedprox_quick] rank log redirected: rank={rank}, local_rank={local_rank}, "
            f"path={log_path}, tee={tee}",
            flush=True,
        )

    # Reproducibility setup for benchmark runs.
    _maybe_redirect_rank_logs(exp_args.training_args.output_dir)
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    if not exp_args.data_args.data_path or not os.path.isdir(exp_args.data_args.data_path):
        raise FileNotFoundError(
            "Training data_path does not exist. "
            f"Resolved path: {exp_args.data_args.data_path!r}. "
            "Please edit exp_args.data_args.data_path in this config file."
        )

    missing_clients = []
    for i in range(exp_args.fed_args.num_clients):
        fp = os.path.join(exp_args.data_args.data_path, f"client_{i}.json")
        if not os.path.exists(fp):
            missing_clients.append(fp)
    if missing_clients:
        raise FileNotFoundError(
            "Missing client shard files for federated training:\n"
            + "\n".join(missing_clients[:20])
        )

    if exp_args.eval_args.eval_data_path and not os.path.exists(exp_args.eval_args.eval_data_path):
        print(
            "[minicpmv_crisismmid_fedprox_quick] WARNING: eval_data_path not found: "
            f"{exp_args.eval_args.eval_data_path}. Per-round eval will be skipped."
        )

    print(
        "[minicpmv_crisismmid_fedprox_quick] data_path="
        f"{exp_args.data_args.data_path}, eval_data_path={exp_args.eval_args.eval_data_path}"
    )
    print(
        "[minicpmv_crisismmid_fedprox_quick] profile="
        f"(rounds={exp_args.fed_args.num_rounds}, max_steps={exp_args.training_args.max_steps}, "
        f"batch={exp_args.training_args.per_device_train_batch_size}, grad_acc={exp_args.training_args.gradient_accumulation_steps})"
    )
    if exp_args.training_args.max_steps > 0:
        per_round_samples = (
            exp_args.training_args.max_steps
            * exp_args.training_args.per_device_train_batch_size
            * exp_args.training_args.gradient_accumulation_steps
        )
        print(
            "[minicpmv_crisismmid_fedprox_quick] NOTE: max_steps>0 means each client uses at most "
            f"{per_round_samples} samples/round (before dataloader short-batch effects)."
        )

    class PrintMetricsHook(FederatedHook):
        """
        杞婚噺绾ф寚鏍囨墦鍗?Hook銆?        姣忚疆缁撴潫鍚庢墦鍗帮細
          - 杞銆佸悇瀹㈡埛绔?loss
          - GPU 鏄惧瓨鍗犵敤锛堣嫢鏈?CUDA锛?          - 璇勪及 Acc / F1锛堣嫢閰嶇疆浜?eval_data_path 涓旀枃浠跺瓨鍦級
        涓嶄緷璧栧鏉傜殑 EvalHook 绯荤粺銆?        """
        def __init__(self, eval_args, output_dir):
            self._ea = eval_args
            self._output_dir = output_dir
            self._round_losses = {}   # {round_idx: [(client, loss)]}
            self._clients_this_round = {}  # {round_idx: [client_ids]}
            self._last_round_idx = -1
            self._model = None
            self._tokenizer = None
            self._eval_ready = False
            self._total_comm_bytes = 0    # 累计通信量（字节）
            self._round_start_ts = {}     # {round_idx: ts}
            self._round_wall_time = {}    # {round_idx: seconds}
            self._eval_history = []       # list of eval metric records

            import os
            ep = getattr(eval_args, "eval_data_path", None)
            if ep and os.path.exists(ep):
                self._eval_data_path = ep
                self._eval_ready = True
            else:
                self._eval_data_path = None
                print("[PrintMetricsHook] eval_data_path not found; skip per-round evaluation.")

        def on_train_start(self, context):
            self._model = context.get("model")
            self._tokenizer = context.get("tokenizer")
            if self._eval_ready and self._model is not None and getattr(self._ea, "startup_eval_full", True):
                # Baseline on the full test set before any federated updates.
                print("[Startup Full Eval] Begin full-test evaluation before round-1 training.", flush=True)
                self._run_eval(round_idx=-1, force_full_eval=True, eval_tag="Startup Full Eval")

        def on_round_start(self, context):
            r = context.get("round_idx", -1)
            self._clients_this_round[r] = context.get("clients_this_round", [])
            self._round_start_ts[r] = time.time()

        def on_client_end(self, context):
            r = context.get("round_idx", -1)
            loss = context.get("training_loss", float("nan"))
            client = context.get("client_idx", -1)
            self._round_losses.setdefault(r, []).append((client, loss))

        def on_aggregate_end(self, context):
            r = context.get("round_idx", -1)
            self._last_round_idx = r
            losses = self._round_losses.get(r, [])

            # 鈹€鈹€ 鎵撳嵃 loss 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            loss_str = "  ".join(
                f"client{c}={l:.6g}" for c, l in sorted(losses)
            )
            print(f"\n{'='*60}")
            print(f"[Round {r+1}] Loss 鈫?{loss_str if loss_str else 'N/A'}")

            # 鈹€鈹€ 鎵撳嵃 GPU 鏄惧瓨 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if torch.cuda.is_available():
                alloc  = torch.cuda.memory_allocated()  / 1024**3
                reserv = torch.cuda.memory_reserved()   / 1024**3
                print(f"[Round {r+1}] GPU  鈫?allocated={alloc:.2f}GB  reserved={reserv:.2f}GB")

            # 鈹€鈹€ 閫氫俊閲忕粺璁?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            # 鍏紡锛氭瘡杞€氫俊 = 2(涓婁紶+涓嬭浇) 脳 鍙備笌瀹㈡埛绔暟 脳 LoRA鍙傛暟閲?脳 2瀛楄妭(bf16)
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
                f"[Round {r+1}] Comm 鈫?"
                f"鏈疆={_fmt(round_comm_bytes)}  "
                f"绱={_fmt(self._total_comm_bytes)}  "
                f"(LoRA鍙傛暟 {param_count:,}  脳  {n_clients}瀹㈡埛绔?"
            )
            t0 = self._round_start_ts.get(r)
            if t0 is not None:
                dt = max(0.0, time.time() - t0)
                self._round_wall_time[r] = dt
                print(f"[Round {r+1}] Time 鈫?{dt:.2f}s")

            # 鈹€鈹€ 鍙€夛細姣忚疆璇勪及 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if self._eval_ready and self._model is not None:
                freq = getattr(self._ea, "eval_freq", 1)
                if (r + 1) % freq == 0:
                    self._run_eval(r)
            print(f"{'='*60}\n")

        def on_train_end(self, context):
            # Final benchmark metric: always run one full-set evaluation at the end.
            if self._eval_ready and self._model is not None and getattr(self._ea, "final_eval_full", False):
                self._run_eval(self._last_round_idx, force_full_eval=True)
            rounds = self._last_round_idx + 1
            round_times = list(self._round_wall_time.values())
            avg_round_time = (sum(round_times) / len(round_times)) if round_times else float("nan")
            def _fmt(b):
                if b >= 1024**3: return f"{b/1024**3:.3f} GB"
                if b >= 1024**2: return f"{b/1024**2:.3f} MB"
                return f"{b/1024:.3f} KB"
            print(f"\n{'='*60}")
            print("[Benchmark Summary]")
            print(f"Rounds completed: {max(rounds, 0)}")
            if round_times:
                print(f"Avg round time: {avg_round_time:.2f}s")
            print(f"Total communication: {_fmt(self._total_comm_bytes)}")
            if self._eval_history:
                best_acc = max(self._eval_history, key=lambda x: x["accuracy"])
                best_f1w = max(self._eval_history, key=lambda x: x["f1_weighted"])
                auc_history = [x for x in self._eval_history if x.get("auc_ovr_macro") is not None]
                print(
                    f"Best Acc: {best_acc['accuracy']:.4f} "
                    f"(round={best_acc['round_idx'] + 1}, scope={best_acc['eval_scope']})"
                )
                print(
                    f"Best F1-W: {best_f1w['f1_weighted']:.4f} "
                    f"(round={best_f1w['round_idx'] + 1}, scope={best_f1w['eval_scope']})"
                )
                if auc_history:
                    best_auc = max(auc_history, key=lambda x: x["auc_ovr_macro"])
                    print(
                        f"Best AUC-M: {best_auc['auc_ovr_macro']:.4f} "
                        f"(round={best_auc['round_idx'] + 1}, scope={best_auc['eval_scope']})"
                    )
                else:
                    print("Best AUC-M: N/A")
            else:
                print("Eval metrics: N/A (no eval rounds executed)")
            print(f"{'='*60}\n")

        def _run_eval(self, round_idx, force_full_eval=False, eval_tag=None):
            import csv
            import json
            import os
            import random
            from collections import Counter
            from fling_mllm.utils.eval_utils import (
                correct_image_paths, extract_question, extract_ground_truth,
                generate_answer, generate_label_probability_scores, match_prediction_to_label,
                compute_classification_metrics, normalize_label,
                classify_prediction_error, summarize_prediction_errors,
                log_first_eval_sample_snapshot,
            )
            with open(self._eval_data_path, "r", encoding="utf-8") as f:
                full_test_data = json.load(f)
            full_test_data = correct_image_paths(full_test_data, os.getcwd())
            full_gts = [extract_ground_truth(s) for s in full_test_data]
            full_label_set = sorted(set(full_gts))
            max_s = None if force_full_eval else getattr(self._ea, "max_samples", None)
            sampled_eval = False
            test_data = full_test_data
            if max_s is not None and max_s > 0 and len(full_test_data) > max_s:
                sample_seed = int(getattr(self._ea, "eval_sample_seed", 42))
                rng = random.Random(sample_seed)
                test_data = rng.sample(full_test_data, max_s)
                sampled_eval = True
            label_set = full_label_set
            if eval_tag is None:
                eval_tag = "Final Eval" if force_full_eval else f"Round {round_idx+1} Eval"
            if round_idx < 0 and force_full_eval:
                eval_scope = "startup_full"
            else:
                eval_scope = "full" if force_full_eval else "sampled"

            # ---- Eval dataset diagnostics ----
            eval_gts_preview = [extract_ground_truth(s) for s in test_data]
            full_gt_dist = Counter(full_gts)
            eval_gt_dist = Counter(eval_gts_preview)
            print(
                f"[{eval_tag}] Running on {len(test_data)} samples "
                f"(full_test={len(full_test_data)}, scope={eval_scope}, sampled={sampled_eval})",
                flush=True,
            )
            print(f"[{eval_tag}] label_set={label_set}", flush=True)
            print(f"[{eval_tag}] GT distribution on full test:", flush=True)
            for lbl in label_set:
                print(f"  - {lbl}: {full_gt_dist.get(lbl, 0)}", flush=True)
            print(f"[{eval_tag}] GT distribution on current eval subset:", flush=True)
            for lbl in label_set:
                print(f"  - {lbl}: {eval_gt_dist.get(lbl, 0)}", flush=True)
            if sampled_eval:
                missing_subset_labels = [lbl for lbl in label_set if eval_gt_dist.get(lbl, 0) == 0]
                if missing_subset_labels:
                    print(
                        f"[{eval_tag}] WARNING: sampled eval missing labels: {missing_subset_labels}",
                        flush=True,
                    )
            if len(test_data) > 0:
                log_first_eval_sample_snapshot(
                    sample=test_data[0],
                    tokenizer=self._tokenizer,
                    model=self._model,
                    stage_tag=eval_tag,
                )

            self._model.eval()
            preds, gts, raw_outputs, score_matrix = [], [], [], []
            unknown_aliases = {"unk", "unknown", "default", "other", "others", "na", "n a", "none"}
            norm_label_set = {normalize_label(lbl) for lbl in label_set}
            empty_raw_count = 0
            parse_failed_count = 0
            parsed_oos_count = 0
            fallback_mapped_count = 0
            unknown_pred_count = 0
            per_sample_records = []
            import torch as _torch
            for i, sample in enumerate(test_data):
                q = extract_question(sample)
                gt = extract_ground_truth(sample)
                img = sample.get("image")
                try:
                    with _torch.no_grad():
                        raw = generate_answer(
                            model=self._model,
                            tokenizer=self._tokenizer,
                            question=q,
                            image_path=img,
                            max_new_tokens=getattr(self._ea, "max_new_tokens", 32),
                        )
                except Exception:
                    raw = ""
                parsed = match_prediction_to_label(raw, label_set, question=q)
                scores = generate_label_probability_scores(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    question=q,
                    label_set=label_set,
                    image_path=img,
                    max_new_tokens=getattr(self._ea, "score_max_new_tokens", 192),
                    fallback_text=raw,
                )
                raw_outputs.append(raw)
                preds.append(parsed)
                score_matrix.append(scores)
                gts.append(gt)

                raw_norm = normalize_label(raw)
                parsed_norm = normalize_label(parsed)
                exact_raw_in_label_set = bool(raw_norm) and (raw_norm in norm_label_set)
                in_label_set = parsed in label_set

                if not str(raw).strip():
                    empty_raw_count += 1
                if not in_label_set:
                    parse_failed_count += 1
                    parsed_oos_count += 1
                if in_label_set and not exact_raw_in_label_set:
                    fallback_mapped_count += 1
                if parsed_norm in unknown_aliases:
                    unknown_pred_count += 1

                topk_scores = []
                if isinstance(scores, (list, tuple)) and len(scores) == len(label_set):
                    ranking = sorted(
                        [(label_set[j], float(scores[j])) for j in range(len(label_set))],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    topk_scores = ranking[: int(getattr(self._ea, "score_topk", 3))]

                sample_id = sample.get("id", sample.get("sample_id", f"idx_{i}"))
                error_type = classify_prediction_error(raw, parsed, gt)
                per_sample_records.append({
                    "round_idx": int(round_idx),
                    "eval_scope": eval_scope,
                    "sample_id": sample_id,   # keep for backward compatibility
                    "id": sample_id,
                    "image_path": img,
                    "question": q,
                    "ground_truth": gt,       # keep for backward compatibility
                    "gt": gt,
                    "raw_output": raw,
                    "raw_output_repr": repr(raw),
                    "parsed_prediction": parsed,  # keep for backward compatibility
                    "parsed_pred": parsed,
                    "correct": bool(parsed == gt),
                    "error_type": error_type,
                    "topk_scores": topk_scores,
                    "score_vector": [float(x) for x in scores] if isinstance(scores, (list, tuple)) else None,
                })
                print(
                    f"[{eval_tag}][PredLog] "
                    f"{json.dumps({'id': sample_id, 'gt': gt, 'raw_output': raw, 'raw_output_repr': repr(raw), 'parsed_pred': parsed, 'correct': bool(parsed == gt), 'error_type': error_type}, ensure_ascii=False)}",
                    flush=True,
                )
            self._model.train()

            # ---- Consistency checks ----
            gt_oos = sorted({g for g in gts if g not in label_set})
            pred_oos = sorted({p for p in preds if p not in label_set})
            if gt_oos:
                print(f"[{eval_tag}] WARNING: GT labels out of label_set: {gt_oos}", flush=True)
            if pred_oos:
                print(f"[{eval_tag}] WARNING: parsed predictions out of label_set: {pred_oos}", flush=True)
            letter_like_labels = [lbl for lbl in label_set if len(lbl.strip()) == 1 and lbl.strip().isalpha()]
            if 0 < len(letter_like_labels) < len(label_set):
                print(
                    f"[{eval_tag}] WARNING: label_set mixes letter-style and long labels; check mapping consistency.",
                    flush=True,
                )

            m = compute_classification_metrics(preds, gts, label_set, score_matrix=score_matrix)
            auc_macro = m.get("auc_ovr_macro")
            auc_str = f"{auc_macro:.4f}" if auc_macro is not None else "N/A"
            auc_note = m.get("auc_note", "auc_note_unset")
            auc_rows = f"{m.get('auc_valid_rows', 0)}/{m.get('auc_total_rows', 0)}"
            auc_cls = m.get("auc_present_classes", 0)
            auc_skip = m.get("auc_skipped_rows", {})
            print(
                f"[{eval_tag}] "
                f"Acc={m['accuracy']:.4f}  "
                f"F1-W={m['f1_weighted']:.4f}  "
                f"F1-M={m['f1_macro']:.4f}  "
                f"AUC-M={auc_str}  "
                f"(n={m['num_samples']}, auc_rows={auc_rows}, auc_cls={auc_cls}, auc_note={auc_note})",
                flush=True
            )
            print(
                f"[{eval_tag}] AUC-Diag 鈫?"
                f"present_labels={m.get('auc_present_labels', [])}  "
                f"skipped={auc_skip}",
                flush=True,
            )
            err = summarize_prediction_errors(per_sample_records)
            total = max(1, err["total"])
            print(f"[{eval_tag}] Error Analysis", flush=True)
            print(f"Total: {err['total']}", flush=True)
            print(f"Correct: {err['correct']}", flush=True)
            print(
                f"Parse Error: {err['parse_error']} ({err['parse_error'] / total:.2%})",
                flush=True,
            )
            print(
                f"Empty Output: {err['empty_output']} ({err['empty_output'] / total:.2%})",
                flush=True,
            )
            print(
                f"Wrong Prediction: {err['wrong_prediction']} ({err['wrong_prediction'] / total:.2%})",
                flush=True,
            )

            # ---- Prediction distribution diagnostics ----
            pred_dist = Counter(preds)
            gt_dist = Counter(gts)
            print(f"[{eval_tag}] Prediction distribution vs GT:", flush=True)
            for lbl in label_set:
                print(
                    f"  - {lbl}: pred={pred_dist.get(lbl, 0)} | gt={gt_dist.get(lbl, 0)}",
                    flush=True,
                )
            never_predicted = [lbl for lbl in label_set if pred_dist.get(lbl, 0) == 0]
            if never_predicted:
                print(f"[{eval_tag}] WARNING: labels never predicted: {never_predicted}", flush=True)

            # ---- Per-class metrics diagnostics ----
            per_class = m.get("per_class_report", {})
            print(f"[{eval_tag}] Per-class metrics:", flush=True)
            for lbl in label_set:
                item = per_class.get(lbl, {})
                print(
                    "  - "
                    f"{lbl}: support={int(item.get('support', 0))} "
                    f"precision={float(item.get('precision', 0.0)):.4f} "
                    f"recall={float(item.get('recall', 0.0)):.4f} "
                    f"f1={float(item.get('f1-score', 0.0)):.4f}",
                    flush=True,
                )

            # ---- Confusion matrix diagnostics ----
            cm = m.get("confusion_matrix", [])
            print(f"[{eval_tag}] Confusion matrix (rows=GT, cols=Pred):", flush=True)
            print(f"  labels={label_set}", flush=True)
            for lbl, row in zip(label_set, cm):
                print(f"  {lbl}: {row}", flush=True)

            # ---- Abnormal outputs diagnostics ----
            n_eval = max(1, len(test_data))
            print(
                f"[{eval_tag}] Abnormal outputs: "
                f"raw_empty={empty_raw_count} ({empty_raw_count/n_eval:.2%}), "
                f"parse_failed_or_oos={parse_failed_count} ({parse_failed_count/n_eval:.2%}), "
                f"parsed_oos={parsed_oos_count} ({parsed_oos_count/n_eval:.2%}), "
                f"fallback_mapped={fallback_mapped_count} ({fallback_mapped_count/n_eval:.2%}), "
                f"unknown_like={unknown_pred_count} ({unknown_pred_count/n_eval:.2%})",
                flush=True,
            )

            # ---- Sample-level debug prints ----
            sample_print_n = int(getattr(self._ea, "debug_print_samples", 10))
            if sample_print_n > 0 and per_sample_records:
                rng = random.Random(int(getattr(self._ea, "eval_sample_seed", 42)) + round_idx + 1000)
                shown = rng.sample(per_sample_records, min(sample_print_n, len(per_sample_records)))
                print(f"[{eval_tag}] Sample-level predictions ({len(shown)} shown):", flush=True)
                for rec in shown:
                    topk = rec.get("topk_scores") or []
                    q_max_chars = int(getattr(self._ea, "debug_question_max_chars", 220))
                    q_text = str(rec["question"])
                    q_preview = q_text if q_max_chars <= 0 else q_text[:q_max_chars]
                    print(
                        f"  - id={rec['sample_id']} correct={rec['correct']} "
                        f"gt={rec['ground_truth']} pred={rec['parsed_prediction']}",
                        flush=True,
                    )
                    print(f"    raw={rec['raw_output']!r}", flush=True)
                    print(f"    q_len={len(q_text)} q={q_preview!r}", flush=True)
                    if topk:
                        print(f"    topk_scores={topk}", flush=True)
                    else:
                        print("    topk_scores=N/A", flush=True)

            # ---- Persist detailed eval artifacts ----
            os.makedirs(self._output_dir, exist_ok=True)
            round_tag = "startup" if round_idx < 0 else f"{round_idx + 1}"
            cm_json_path = os.path.join(self._output_dir, f"confusion_matrix_round_{round_tag}_{eval_scope}.json")
            cm_csv_path = os.path.join(self._output_dir, f"confusion_matrix_round_{round_tag}_{eval_scope}.csv")
            with open(cm_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round_idx": int(round_idx),
                        "eval_scope": eval_scope,
                        "label_set": label_set,
                        "confusion_matrix": cm,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with open(cm_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["gt\\pred"] + label_set)
                for lbl, row in zip(label_set, cm):
                    writer.writerow([lbl] + list(row))

            pred_jsonl_path = os.path.join(self._output_dir, f"eval_predictions_round_{round_tag}_{eval_scope}.jsonl")
            with open(pred_jsonl_path, "w", encoding="utf-8") as f:
                for rec in per_sample_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Unified rolling prediction log for quick-debug workflow.
            merged_pred_path = os.path.join(self._output_dir, "output_predictions.jsonl")
            with open(merged_pred_path, "a", encoding="utf-8") as f:
                for rec in per_sample_records:
                    f.write(
                        json.dumps(
                            {
                                "id": rec["id"],
                                "gt": rec["gt"],
                                "raw_output": rec["raw_output"],
                                "raw_output_repr": rec["raw_output_repr"],
                                "parsed_pred": rec["parsed_pred"],
                                "correct": rec["correct"],
                                "error_type": rec["error_type"],
                                "round_idx": rec["round_idx"],
                                "eval_scope": rec["eval_scope"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            diagnostics_path = os.path.join(self._output_dir, f"eval_diagnostics_round_{round_tag}_{eval_scope}.json")
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round_idx": int(round_idx),
                        "eval_scope": eval_scope,
                        "label_set": label_set,
                        "full_test_num_samples": len(full_test_data),
                        "eval_num_samples": len(test_data),
                        "full_gt_distribution": {k: int(full_gt_dist.get(k, 0)) for k in label_set},
                        "eval_gt_distribution": {k: int(gt_dist.get(k, 0)) for k in label_set},
                        "pred_distribution": {k: int(pred_dist.get(k, 0)) for k in label_set},
                        "abnormal_output_stats": {
                            "raw_empty_count": int(empty_raw_count),
                            "parse_failed_or_oos_count": int(parse_failed_count),
                            "parsed_oos_count": int(parsed_oos_count),
                            "fallback_mapped_count": int(fallback_mapped_count),
                            "unknown_like_count": int(unknown_pred_count),
                        },
                        "metrics": m,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Keep original per-round metrics history semantics unchanged.
            rec = {"round_idx": round_idx, "eval_scope": eval_scope, **{k: m[k] for k in
                   ("accuracy", "f1_weighted", "f1_macro", "auc_ovr_macro", "auc_ovr_weighted",
                    "auc_note", "auc_valid_rows", "auc_total_rows", "auc_present_classes",
                    "auc_present_labels", "auc_skipped_rows", "num_samples")}}
            self._eval_history.append(rec)
            with open(os.path.join(self._output_dir, "eval_metrics_per_round.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    hooks = [PrintMetricsHook(exp_args.eval_args, exp_args.training_args.output_dir)]
    exp_args.hooks = hooks
    try:
        run_federated_finetune_from_config(exp_args)
    except Exception as e:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(f"[FATAL][rank={rank} local_rank={local_rank}] {e}")
        traceback.print_exc()
        raise

