import json
import os
import random
import time
import traceback

import numpy as np
import torch

from fling_mllm.federated.hooks import FederatedHook
from fling_mllm.pipeline import run_federated_finetune_from_config
from fling_mllm.tasks import (
    build_task_evaluator,
    build_task_loader_kwargs,
    normalize_task_type,
    resolve_client_data_path,
)


def run_federated_with_task_metrics_hook(exp_args, run_name: str):
    def _maybe_redirect_rank_logs(output_dir):
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
        try:
            resolve_client_data_path(exp_args.data_args.data_path, i)
        except FileNotFoundError as exc:
            missing_clients.append(str(exc))
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

    task_type = normalize_task_type(getattr(exp_args.data_args, "task_type", "classification"))
    print(
        f"[{run_name}] task_type={task_type}, data_path={exp_args.data_args.data_path}, "
        f"eval_data_path={exp_args.eval_args.eval_data_path}"
    )

    class PrintMetricsHook(FederatedHook):
        def __init__(self, eval_args, data_args, output_dir):
            self._ea = eval_args
            self._da = data_args
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
            self._evaluator = None

            self._task_type = normalize_task_type(
                getattr(eval_args, "task_type", None)
                or getattr(data_args, "task_type", "classification")
            )
            self._data_format = getattr(eval_args, "data_format", None) or getattr(data_args, "data_format", "auto")
            self._eval_split = getattr(eval_args, "eval_split", None) or getattr(data_args, "eval_split", "eval")
            self._loader_kwargs = build_task_loader_kwargs(task_type=self._task_type, data_args=data_args)
            if self._task_type == "vqa":
                if getattr(eval_args, "vqa_image_root", None) is not None:
                    self._loader_kwargs["vqa_image_root"] = getattr(eval_args, "vqa_image_root")
                if getattr(eval_args, "vqa_prompt_template", None) is not None:
                    self._loader_kwargs["vqa_prompt_template"] = getattr(eval_args, "vqa_prompt_template")
                if getattr(eval_args, "strict_image_path", None) is not None:
                    self._loader_kwargs["strict_image_path"] = getattr(eval_args, "strict_image_path")

            ep = getattr(eval_args, "eval_data_path", None)
            if ep and os.path.exists(ep):
                self._eval_data_path = ep
                self._eval_ready = True
                self._evaluator = build_task_evaluator(
                    task_type=self._task_type,
                    eval_data_path=self._eval_data_path,
                    data_format=self._data_format,
                    split=self._eval_split,
                    loader_kwargs=self._loader_kwargs,
                )
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
            self._round_losses.setdefault(r, []).append((client, loss))

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
                self._print_best_eval_summary()
            else:
                print("Eval metrics: N/A (no eval rounds executed)")
            print(f"{'=' * 60}\n")

        def _print_best_eval_summary(self):
            if self._task_type == "vqa":
                best_vqa = max(self._eval_history, key=lambda x: x.get("vqa_score", float("-inf")))
                best_em = max(self._eval_history, key=lambda x: x.get("normalized_exact_match", float("-inf")))
                print(
                    f"Best VQA Score: {best_vqa.get('vqa_score', 0.0):.4f} "
                    f"(round={best_vqa['round_idx'] + 1}, scope={best_vqa['eval_scope']})"
                )
                print(
                    f"Best N-EM: {best_em.get('normalized_exact_match', 0.0):.4f} "
                    f"(round={best_em['round_idx'] + 1}, scope={best_em['eval_scope']})"
                )
                return

            best_acc = max(self._eval_history, key=lambda x: x.get("accuracy", float("-inf")))
            best_f1w = max(self._eval_history, key=lambda x: x.get("f1_weighted", float("-inf")))
            print(
                f"Best Acc: {best_acc.get('accuracy', 0.0):.4f} "
                f"(round={best_acc['round_idx'] + 1}, scope={best_acc['eval_scope']})"
            )
            print(
                f"Best F1-W: {best_f1w.get('f1_weighted', 0.0):.4f} "
                f"(round={best_f1w['round_idx'] + 1}, scope={best_f1w['eval_scope']})"
            )

        def _run_eval(self, round_idx, force_full_eval=False, eval_tag=None):
            if self._evaluator is None:
                return

            if force_full_eval:
                max_s = None
            else:
                max_s = int(getattr(self._ea, "max_samples", 500))

            eval_scope = "full" if force_full_eval else "sampled"
            if round_idx < 0 and force_full_eval:
                eval_scope = "startup_full"

            if eval_tag is None:
                eval_tag = "Final Eval" if force_full_eval else f"Round {round_idx + 1} Eval"

            sample_seed = int(getattr(self._ea, "eval_sample_seed", 42))
            samples = self._evaluator.sample_eval_subset(
                max_samples=max_s,
                sample_seed=sample_seed,
                force_full_eval=force_full_eval,
            )
            print(f"[{eval_tag}] Running on {len(samples)} samples (scope={eval_scope}) ...", flush=True)

            self._model.eval()
            metrics, records = self._evaluator.evaluate(
                model=self._model,
                tokenizer=self._tokenizer,
                samples=samples,
                max_new_tokens=getattr(self._ea, "max_new_tokens", 16),
                device=getattr(self._ea, "device", "cuda"),
                stage_tag=eval_tag,
            )
            self._model.train()

            if self._task_type == "vqa":
                print(
                    f"[{eval_tag}] "
                    f"N-EM={metrics.get('normalized_exact_match', 0.0):.4f}  "
                    f"VQA={metrics.get('vqa_score', 0.0):.4f}  "
                    f"Empty={metrics.get('empty_prediction_rate', 0.0):.4f}  "
                    f"(n={metrics.get('num_samples', 0)})",
                    flush=True,
                )
            else:
                auc_macro = metrics.get("auc_ovr_macro")
                auc_str = f"{auc_macro:.4f}" if auc_macro is not None else "N/A"
                print(
                    f"[{eval_tag}] "
                    f"Acc={metrics.get('accuracy', 0.0):.4f}  "
                    f"F1-W={metrics.get('f1_weighted', 0.0):.4f}  "
                    f"F1-M={metrics.get('f1_macro', 0.0):.4f}  "
                    f"AUC-M={auc_str}  "
                    f"(n={metrics.get('num_samples', 0)})",
                    flush=True,
                )

            os.makedirs(self._output_dir, exist_ok=True)
            pred_path = os.path.join(self._output_dir, "output_predictions.jsonl")
            with open(pred_path, "a", encoding="utf-8") as f:
                for rec in records:
                    out = dict(rec)
                    out["round_idx"] = int(round_idx)
                    out["eval_scope"] = eval_scope
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

            rec = {
                "round_idx": int(round_idx),
                "eval_scope": eval_scope,
                **metrics,
            }
            self._eval_history.append(rec)
            with open(os.path.join(self._output_dir, "eval_metrics_per_round.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hooks = [PrintMetricsHook(exp_args.eval_args, exp_args.data_args, exp_args.training_args.output_dir)]
    exp_args.hooks = hooks
    try:
        run_federated_finetune_from_config(exp_args)
    except Exception as exc:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(f"[FATAL][rank={rank} local_rank={local_rank}] {exc}")
        traceback.print_exc()
        raise
