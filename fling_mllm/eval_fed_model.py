#!/usr/bin/env python3
"""
Standalone evaluation script for federated multimodal checkpoints.

Supports both classification and VQA tasks via task_type.
"""

import argparse
import os
import sys

import torch

# Allow running from project root without install
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from fling_mllm.tasks import (
    build_task_evaluator,
    normalize_task_type,
    save_task_eval_outputs,
)
from fling_mllm.utils.eval_utils import load_model_for_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a federated MLLM checkpoint")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to the saved checkpoint (adapter or full model)")
    parser.add_argument("--test_data_path", required=True,
                        help="Evaluation dataset path")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write evaluation outputs")
    parser.add_argument("--model_name_or_path", default="openbmb/MiniCPM-V-2_6-int4",
                        help="Base model HuggingFace id or local path")
    parser.add_argument("--cache_dir", default=None,
                        help="Model cache directory")
    parser.add_argument("--task_type", default="classification",
                        choices=["classification", "vqa"],
                        help="Task type evaluator")
    parser.add_argument("--data_format", default="auto",
                        help="Data format hint for task loader")
    parser.add_argument("--eval_split", default="eval",
                        help="Eval split name passed to task loader")
    parser.add_argument("--vqa_image_root", default=None,
                        help="VQA image root directory (for VQAv2 layouts)")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max tokens to generate per sample")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu", "auto"],
                        help="Inference device")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit eval sample count")
    parser.add_argument("--eval_sample_seed", type=int, default=42,
                        help="Random seed for sampled evaluation subset")
    return parser.parse_args()


def _print_metric_summary(task_type: str, metrics: dict):
    print("\n" + "=" * 60)
    if task_type == "vqa":
        print(f"  N-EM:         {metrics.get('normalized_exact_match', 0.0):.4f}")
        print(f"  VQA Score:    {metrics.get('vqa_score', 0.0):.4f}")
        print(f"  Empty Rate:   {metrics.get('empty_prediction_rate', 0.0):.4f}")
        print(f"  Num Samples:  {metrics.get('num_samples', 0)}")
    else:
        print(f"  Accuracy:     {metrics.get('accuracy', 0.0):.4f}")
        print(f"  F1 (Weighted):{metrics.get('f1_weighted', 0.0):.4f}")
        print(f"  F1 (Macro):   {metrics.get('f1_macro', 0.0):.4f}")
        auc = metrics.get('auc_ovr_macro')
        print(f"  AUC (OvR-M):  {auc if auc is not None else 'N/A'}")
        print(f"  Num Samples:  {metrics.get('num_samples', 0)}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA not available, falling back to CPU.")
        device = "cpu"

    task_type = normalize_task_type(args.task_type)

    print(f"[eval] Loading model from checkpoint: {args.checkpoint_dir}")
    model, tokenizer = load_model_for_eval(
        checkpoint_dir=args.checkpoint_dir,
        base_model_name=args.model_name_or_path,
        cache_dir=args.cache_dir,
        device=device,
    )
    print("[eval] Model loaded.")

    loader_kwargs = {}
    if task_type == "vqa" and args.vqa_image_root:
        loader_kwargs["vqa_image_root"] = args.vqa_image_root

    evaluator = build_task_evaluator(
        task_type=task_type,
        eval_data_path=args.test_data_path,
        data_format=args.data_format,
        split=args.eval_split,
        loader_kwargs=loader_kwargs,
    )

    samples = evaluator.sample_eval_subset(
        max_samples=args.max_samples,
        sample_seed=args.eval_sample_seed,
        force_full_eval=(args.max_samples is None),
    )
    print(f"[eval] Loaded {len(samples)} samples for task={task_type}.")

    metrics, records = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        max_new_tokens=args.max_new_tokens,
        device=device,
        stage_tag="Eval",
    )

    save_task_eval_outputs(output_dir=args.output_dir, metrics=metrics, records=records)
    _print_metric_summary(task_type=task_type, metrics=metrics)
    print("[eval] Done.")
    return metrics


if __name__ == "__main__":
    main()
