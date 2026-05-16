import os
import torch
import json


class FederatedHook:
    def on_train_start(self, context):
        return None

    def on_round_start(self, context):
        return None

    def on_client_start(self, context):
        return None

    def on_client_end(self, context):
        return None

    def on_aggregate_end(self, context):
        return None

    def on_round_end(self, context):
        return None

    def on_train_end(self, context):
        return None


class FederatedHookManager:
    def __init__(self, hooks=None):
        self.hooks = hooks or []

    def emit(self, event, context):
        for hook in self.hooks:
            fn = getattr(hook, event, None)
            if fn is not None:
                fn(context)


class FederatedTraceHook(FederatedHook):
    def __init__(self, output_path):
        self.output_path = output_path

    def _write(self, event, context):
        record = {"event": event, "context": context}
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_start(self, context):
        self._write("on_train_start", context)

    def on_round_start(self, context):
        self._write("on_round_start", context)

    def on_client_start(self, context):
        self._write("on_client_start", context)

    def on_client_end(self, context):
        self._write("on_client_end", context)

    def on_aggregate_end(self, context):
        self._write("on_aggregate_end", context)

    def on_round_end(self, context):
        self._write("on_round_end", context)

    def on_train_end(self, context):
        self._write("on_train_end", context)


def summarize_state_dict(state_dict):
    tensor_count = 0
    param_count = 0
    l2_norm = 0.0
    for value in state_dict.values():
        if torch.is_tensor(value):
            tensor_count += 1
            param_count += value.numel()
            l2_norm += value.float().norm().item() ** 2
    return {
        "tensor_count": tensor_count,
        "param_count": param_count,
        "l2_norm": l2_norm ** 0.5,
    }


class FederatedEvalHook(FederatedHook):
    """
    Evaluates the global model on a held-out test set after each aggregation round.

    Parameters
    ----------
    model           : The PEFT/full model (same object mutated by the pipeline).
    tokenizer       : Matching tokenizer.
    test_data_path  : Path to test.json (same format as client_X.json).
    output_dir      : Directory to write eval_metrics_per_round.jsonl.
    eval_freq       : Evaluate every N rounds (default 1 = every round).
    max_new_tokens  : Max tokens to generate per sample.
    max_samples     : Limit eval samples (None = all). Useful for quick checks.
    project_root    : Used to correct relative image paths (defaults to cwd).
    """

    def __init__(
        self,
        model,
        tokenizer,
        test_data_path: str,
        output_dir: str,
        eval_freq: int = 1,
        max_new_tokens: int = 32,
        score_max_new_tokens: int = 192,
        max_samples=None,
        project_root: str = None,
    ):
        import json as _json
        from fling_mllm.utils.eval_utils import (
            correct_image_paths,
            extract_question,
            extract_ground_truth,
            generate_answer,
            generate_label_probability_scores,
            log_first_eval_sample_snapshot,
            match_prediction_to_label,
            compute_classification_metrics,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.max_new_tokens = max_new_tokens
        self.score_max_new_tokens = score_max_new_tokens
        self.metrics_path = os.path.join(output_dir, "eval_metrics_per_round.jsonl")

        self._enabled = False
        import os as _os
        project_root = project_root or _os.getcwd()

        # Graceful fallback: skip eval if test file is missing or None
        if test_data_path is None or not _os.path.exists(test_data_path):
            print(
                f"[EvalHook] WARNING: test_data_path not found: {test_data_path!r}\n"
                "  In-training evaluation is DISABLED.\n"
                "  Generate test.json with: python mllmzoo/generate_test_split.py"
            )
            return

        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = _json.load(f)
        test_data = correct_image_paths(test_data, project_root)
        if max_samples is not None:
            test_data = test_data[:max_samples]
        self.test_data = test_data
        self.label_set = sorted(set(extract_ground_truth(s) for s in test_data))
        self._enabled = True

        # Keep references to the helper functions
        self._extract_question = extract_question
        self._extract_ground_truth = extract_ground_truth
        self._generate_answer = generate_answer
        self._score_labels = generate_label_probability_scores
        self._log_first_eval_sample_snapshot = log_first_eval_sample_snapshot
        self._match = match_prediction_to_label
        self._compute_metrics = compute_classification_metrics

        print(
            f"[EvalHook] Initialized: {len(self.test_data)} test samples, "
            f"{len(self.label_set)} labels, eval_freq={eval_freq}"
        )

    def on_aggregate_end(self, context):
        import os as _os
        round_idx = context.get("round_idx", -1)
        if (round_idx + 1) % self.eval_freq != 0:
            return

        print(f"\n[EvalHook] Evaluating global model at round {round_idx + 1} ...")
        self.model.eval()
        if len(self.test_data) > 0:
            self._log_first_eval_sample_snapshot(
                sample=self.test_data[0],
                tokenizer=self.tokenizer,
                model=self.model,
                stage_tag="EvalHook",
            )

        predictions, ground_truths, score_matrix = [], [], []
        for sample in self.test_data:
            question = self._extract_question(sample)
            gt = self._extract_ground_truth(sample)
            image_path = sample.get("image", None)
            try:
                with torch.no_grad():
                    raw_pred = self._generate_answer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        question=question,
                        image_path=image_path,
                        max_new_tokens=self.max_new_tokens,
                    )
            except Exception as e:
                print(f"[EvalHook] WARNING inference error: {e}")
                raw_pred = ""
            predictions.append(self._match(raw_pred, self.label_set, question=question))
            score_row = self._score_labels(
                model=self.model,
                tokenizer=self.tokenizer,
                question=question,
                label_set=self.label_set,
                image_path=image_path,
                max_new_tokens=self.score_max_new_tokens,
                fallback_text=raw_pred,
            )
            score_matrix.append(score_row)
            ground_truths.append(gt)

        metrics = self._compute_metrics(
            predictions,
            ground_truths,
            self.label_set,
            score_matrix=score_matrix,
        )
        record = {
            "round_idx": round_idx,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "f1_macro": metrics["f1_macro"],
            "auc_ovr_macro": metrics.get("auc_ovr_macro"),
            "auc_ovr_weighted": metrics.get("auc_ovr_weighted"),
            "auc_note": metrics.get("auc_note"),
            "auc_valid_rows": metrics.get("auc_valid_rows"),
            "auc_total_rows": metrics.get("auc_total_rows"),
            "auc_present_classes": metrics.get("auc_present_classes"),
            "auc_present_labels": metrics.get("auc_present_labels"),
            "auc_skipped_rows": metrics.get("auc_skipped_rows"),
            "num_samples": metrics["num_samples"],
        }
        auc_macro = metrics.get("auc_ovr_macro")
        auc_str = f"{auc_macro:.4f}" if auc_macro is not None else "N/A"
        auc_note = metrics.get("auc_note", "auc_note_unset")
        auc_rows = f"{metrics.get('auc_valid_rows', 0)}/{metrics.get('auc_total_rows', 0)}"
        auc_cls = metrics.get("auc_present_classes", 0)
        auc_skip = metrics.get("auc_skipped_rows", {})
        print(
            f"[EvalHook] Round {round_idx + 1}: "
            f"Acc={metrics['accuracy']:.4f}  "
            f"F1-W={metrics['f1_weighted']:.4f}  "
            f"F1-M={metrics['f1_macro']:.4f}  "
            f"AUC-M={auc_str}  "
            f"(auc_rows={auc_rows}, auc_cls={auc_cls}, auc_note={auc_note})"
        )
        print(
            f"[EvalHook] Round {round_idx + 1} AUC-Diag: "
            f"present_labels={metrics.get('auc_present_labels', [])}  "
            f"skipped={auc_skip}"
        )
        _os.makedirs(self.output_dir, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Put model back to train mode (pipeline resumes training after this hook)
        self.model.train()
