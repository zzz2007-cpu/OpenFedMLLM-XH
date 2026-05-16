import os
import random
import time
import traceback

import numpy as np
import torch

from fling_mllm.federated.hooks import FederatedHook
from fling_mllm.pipeline import run_federated_finetune_from_config


def run_federated_with_metrics_hook(exp_args, run_name: str):
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
            import sys

            class _Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data):
                    for stream in self.streams:
                        stream.write(data)
                    return len(data)

                def flush(self):
                    for stream in self.streams:
                        stream.flush()

            sys.stdout = _Tee(sys.__stdout__, log_fp)
            sys.stderr = _Tee(sys.__stderr__, log_fp)
        else:
            import sys

            sys.stdout = log_fp
            sys.stderr = log_fp
        print(
            f"[{run_name}] rank log redirected: rank={rank}, local_rank={local_rank}, "
            f"path={log_path}, tee={tee}",
            flush=True,
        )

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
            f"[{run_name}] WARNING: eval_data_path not found: "
            f"{exp_args.eval_args.eval_data_path}. Per-round eval will be skipped."
        )

    print(
        f"[{run_name}] data_path={exp_args.data_args.data_path}, "
        f"eval_data_path={exp_args.eval_args.eval_data_path}"
    )

    class PrintMetricsHook(FederatedHook):
        def __init__(self, eval_args, output_dir):
            self._ea = eval_args
            self._output_dir = output_dir
            self._round_losses = {}
            self._clients_this_round = {}
            self._last_round_idx = -1
            self._model = None
            self._tokenizer = None
            self._eval_ready = False
            self._total_comm_bytes = 0
            self._round_start_ts = {}
            self._round_wall_time = {}
            self._eval_history = []

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
            metrics = context.get("metrics") or {}
            self._round_losses.setdefault(r, []).append((client, loss))
            scaffold_corr = metrics.get("scaffold_corr_norm")
            scaffold_match = metrics.get("scaffold_matched_params")
            grad_norm = metrics.get("grad_norm")
            if scaffold_corr is not None or scaffold_match is not None:
                print(
                    f"[SCAFFOLD][ClientMetrics] round={r + 1} client={client} "
                    f"loss={loss:.6f} grad_norm={grad_norm} "
                    f"corr_l2={scaffold_corr} matched_params={scaffold_match}",
                    flush=True,
                )

        def on_aggregate_end(self, context):
            r = context.get("round_idx", -1)
            self._last_round_idx = r
            losses = self._round_losses.get(r, [])

            loss_str = "  ".join(f"client{c}={l:.4f}" for c, l in sorted(losses))
            print(f"\n{'=' * 60}")
            print(f"[Round {r + 1}] Loss -> {loss_str if loss_str else 'N/A'}")

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserv = torch.cuda.memory_reserved() / 1024**3
                print(f"[Round {r + 1}] GPU  -> allocated={alloc:.2f}GB  reserved={reserv:.2f}GB")

            state_summary = context.get("global_state_summary", {})
            param_count = state_summary.get("param_count", 0)
            clients_this_round = self._clients_this_round.get(r, [])
            n_clients = len(clients_this_round) if clients_this_round else 1
            bytes_per_param = 2
            round_comm_bytes = 2 * n_clients * param_count * bytes_per_param
            self._total_comm_bytes += round_comm_bytes

            def _fmt(size_bytes):
                if size_bytes >= 1024**3:
                    return f"{size_bytes / 1024**3:.3f} GB"
                if size_bytes >= 1024**2:
                    return f"{size_bytes / 1024**2:.3f} MB"
                return f"{size_bytes / 1024:.3f} KB"

            print(
                f"[Round {r + 1}] Comm -> this_round={_fmt(round_comm_bytes)}  "
                f"total={_fmt(self._total_comm_bytes)}  "
                f"(LoRA params={param_count:,}, clients={n_clients})"
            )

            t0 = self._round_start_ts.get(r)
            if t0 is not None:
                dt = max(0.0, time.time() - t0)
                self._round_wall_time[r] = dt
                print(f"[Round {r + 1}] Time -> {dt:.2f}s")

            if self._eval_ready and self._model is not None:
                freq = getattr(self._ea, "eval_freq", 1)
                if (r + 1) % freq == 0:
                    self._run_eval(r)
            print(f"{'=' * 60}\n")

        def on_train_end(self, context):
            if self._eval_ready and self._model is not None and getattr(self._ea, "final_eval_full", False):
                self._run_eval(self._last_round_idx, force_full_eval=True)

            rounds = self._last_round_idx + 1
            round_times = list(self._round_wall_time.values())
            avg_round_time = (sum(round_times) / len(round_times)) if round_times else float("nan")

            def _fmt(size_bytes):
                if size_bytes >= 1024**3:
                    return f"{size_bytes / 1024**3:.3f} GB"
                if size_bytes >= 1024**2:
                    return f"{size_bytes / 1024**2:.3f} MB"
                return f"{size_bytes / 1024:.3f} KB"

            print(f"\n{'=' * 60}")
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
            print(f"{'=' * 60}\n")

        def _run_eval(self, round_idx, force_full_eval=False, eval_tag=None):
            import json

            from fling_mllm.utils.eval_utils import (
                classify_prediction_error,
                compute_classification_metrics,
                correct_image_paths,
                extract_ground_truth,
                extract_question,
                generate_answer,
                generate_label_probability_scores,
                log_first_eval_sample_snapshot,
                match_prediction_to_label,
                summarize_prediction_errors,
            )

            with open(self._eval_data_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            test_data = correct_image_paths(test_data, os.getcwd())
            full_label_set = sorted(set(extract_ground_truth(s) for s in test_data))

            if force_full_eval:
                max_s = None
            else:
                max_s = int(getattr(self._ea, "max_samples", 500))

            if max_s is not None and max_s > 0 and len(test_data) > max_s:
                sample_seed = int(getattr(self._ea, "eval_sample_seed", 42))
                rng = random.Random(sample_seed)
                test_data = rng.sample(test_data, max_s)

            label_set = full_label_set
            if eval_tag is None:
                eval_tag = "Final Eval" if force_full_eval else f"Round {round_idx + 1} Eval"
            if force_full_eval:
                print(f"[{eval_tag}] Running on {len(test_data)} samples (full eval) ...", flush=True)
            else:
                print(
                    f"[{eval_tag}] Running on {len(test_data)} samples "
                    f"(fixed max_samples={max_s}) ...",
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
            preds, gts, score_matrix = [], [], []
            per_sample_records = []
            for i, sample in enumerate(test_data):
                q = extract_question(sample)
                gt = extract_ground_truth(sample)
                img = sample.get("image")
                try:
                    with torch.no_grad():
                        raw = generate_answer(
                            model=self._model,
                            tokenizer=self._tokenizer,
                            question=q,
                            image_path=img,
                            max_new_tokens=getattr(self._ea, "max_new_tokens", 32),
                        )
                except Exception:
                    raw = ""
                parsed_pred = match_prediction_to_label(raw, label_set, question=q)
                preds.append(parsed_pred)
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

                sample_id = sample.get("id", sample.get("sample_id", f"idx_{i}"))
                error_type = classify_prediction_error(raw, parsed_pred, gt)
                pred_record = {
                    "id": sample_id,
                    "gt": gt,
                    "raw_output": raw,
                    "raw_output_repr": repr(raw),
                    "parsed_pred": parsed_pred,
                    "correct": bool(parsed_pred == gt),
                    "error_type": error_type,
                    "round_idx": int(round_idx),
                    "eval_scope": "full" if force_full_eval else "sampled",
                }
                per_sample_records.append(pred_record)
                print(f"[{eval_tag}][PredLog] {json.dumps(pred_record, ensure_ascii=False)}", flush=True)

            self._model.train()

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
                flush=True,
            )
            print(
                f"[{eval_tag}] AUC-Diag -> "
                f"present_labels={m.get('auc_present_labels', [])}  "
                f"skipped={auc_skip}",
                flush=True,
            )

            err = summarize_prediction_errors(per_sample_records)
            total = max(1, err["total"])
            print(f"[{eval_tag}] Error Analysis", flush=True)
            print(f"Total: {err['total']}", flush=True)
            print(f"Correct: {err['correct']}", flush=True)
            print(f"Parse Error: {err['parse_error']} ({err['parse_error'] / total:.2%})", flush=True)
            print(f"Empty Output: {err['empty_output']} ({err['empty_output'] / total:.2%})", flush=True)
            print(
                f"Wrong Prediction: {err['wrong_prediction']} ({err['wrong_prediction'] / total:.2%})",
                flush=True,
            )

            os.makedirs(self._output_dir, exist_ok=True)
            pred_path = os.path.join(self._output_dir, "output_predictions.jsonl")
            with open(pred_path, "a", encoding="utf-8") as f:
                for rec in per_sample_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{eval_tag}] Saved prediction logs -> {pred_path}", flush=True)

            if round_idx < 0 and force_full_eval:
                eval_scope = "startup_full"
            else:
                eval_scope = "full" if force_full_eval else "sampled"

            rec = {
                "round_idx": round_idx,
                "eval_scope": eval_scope,
                **{
                    k: m[k]
                    for k in (
                        "accuracy",
                        "f1_weighted",
                        "f1_macro",
                        "auc_ovr_macro",
                        "auc_ovr_weighted",
                        "auc_note",
                        "auc_valid_rows",
                        "auc_total_rows",
                        "auc_present_classes",
                        "auc_present_labels",
                        "auc_skipped_rows",
                        "num_samples",
                    )
                },
            }
            self._eval_history.append(rec)
            with open(os.path.join(self._output_dir, "eval_metrics_per_round.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    hooks = [PrintMetricsHook(exp_args.eval_args, exp_args.training_args.output_dir)]
    exp_args.hooks = hooks
    try:
        run_federated_finetune_from_config(exp_args)
    except Exception as exc:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(f"[FATAL][rank={rank} local_rank={local_rank}] {exc}")
        traceback.print_exc()
        raise
