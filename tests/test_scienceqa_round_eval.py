import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fling_mllm.pipeline import baseline_runner
from fling_mllm.pipeline.baseline_runner import (
    FinalModelEvalHook,
    PerRoundModelEvalHook,
)
from fling_mllm.pipeline.mode_dispatcher import _build_federated_eval_hooks
from mllmzoo.configs.scienceqa_chi._base import build_scienceqa_chi_exp


class _FakeModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeEvaluator:
    def __init__(self, samples):
        self.samples = samples
        self.eval_sample_ids = []

    def sample_eval_subset(self, max_samples, sample_seed, force_full_eval=False):
        if force_full_eval or max_samples is None or len(self.samples) <= max_samples:
            return list(self.samples)
        return random.Random(sample_seed).sample(self.samples, max_samples)

    def evaluate(self, model, tokenizer, samples, **kwargs):
        del model, tokenizer, kwargs
        sample_ids = [sample["id"] for sample in samples]
        self.eval_sample_ids.append(sample_ids)
        metrics = {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "invalid_rate": 0.0,
            "num_samples": len(samples),
        }
        records = [{"id": sample_id} for sample_id in sample_ids]
        return metrics, records


class ScienceQARoundEvalTests(unittest.TestCase):
    def test_four_heterogeneity_configs_share_round_eval_defaults(self):
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"EVAL_EVERY", "EVAL_MAX_SAMPLES", "EVAL_SAMPLE_SEED"}
        }
        settings = ["L0_M0", "L1_M0", "L0_M1", "L1_M1"]
        with mock.patch.dict(os.environ, env, clear=True):
            configs = [
                build_scienceqa_chi_exp("FedAvg-LoRA", setting, setting)
                for setting in settings
            ]

        self.assertTrue(all(config.eval_args.eval_freq == 1 for config in configs))
        self.assertTrue(all(config.eval_args.max_samples == 100 for config in configs))
        self.assertTrue(all(config.eval_args.eval_sample_seed == 42 for config in configs))
        self.assertTrue(all(config.eval_args.final_eval_full for config in configs))
        self.assertEqual(
            {Path(config.eval_args.eval_data_path).name for config in configs},
            {"validation.jsonl"},
        )

    def test_dispatcher_registers_round_hook_before_final_hook(self):
        hooks = _build_federated_eval_hooks(
            existing_hooks=[],
            eval_args={"eval_data_path": "validation.jsonl", "eval_freq": 1},
            output_dir="output",
            enable_final_eval=True,
        )
        self.assertEqual(len(hooks), 2)
        self.assertIsInstance(hooks[0], PerRoundModelEvalHook)
        self.assertIsInstance(hooks[1], FinalModelEvalHook)

    def test_round_hook_reuses_same_100_samples_each_round(self):
        samples = [{"id": f"sample-{idx}"} for idx in range(150)]
        evaluator = _FakeEvaluator(samples)
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = os.path.join(tmpdir, "validation.jsonl")
            Path(eval_path).write_text("{}\n", encoding="utf-8")
            hook = PerRoundModelEvalHook(
                eval_args={
                    "eval_data_path": eval_path,
                    "task_type": "scienceqa",
                    "data_format": "jsonl",
                    "eval_freq": 1,
                    "max_samples": 100,
                    "eval_sample_seed": 42,
                },
                output_dir=tmpdir,
            )
            with mock.patch.object(
                baseline_runner, "build_task_evaluator", return_value=evaluator
            ):
                hook.on_train_start({"model": model, "tokenizer": object()})
                hook.on_aggregate_end({"round_idx": 0})
                hook.on_aggregate_end({"round_idx": 1})

            self.assertEqual(len(evaluator.eval_sample_ids), 2)
            self.assertEqual(len(evaluator.eval_sample_ids[0]), 100)
            self.assertEqual(evaluator.eval_sample_ids[0], evaluator.eval_sample_ids[1])
            self.assertTrue(model.training)

            sample_manifest = json.loads(
                Path(tmpdir, "eval_sample_ids.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sample_manifest["sample_ids"], evaluator.eval_sample_ids[0])
            metric_lines = Path(tmpdir, "eval_metrics_per_round.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(metric_lines), 2)
            self.assertEqual([json.loads(line)["round_idx"] for line in metric_lines], [0, 1])

    def test_final_hook_forces_full_eval_and_writes_best_round(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = os.path.join(tmpdir, "validation.jsonl")
            Path(eval_path).write_text("{}\n", encoding="utf-8")
            per_round_path = Path(tmpdir, "eval_metrics_per_round.jsonl")
            per_round_path.write_text(
                "\n".join(
                    [
                        json.dumps({"round_idx": 0, "accuracy": 0.4, "macro_f1": 0.6}),
                        json.dumps({"round_idx": 1, "accuracy": 0.8, "macro_f1": 0.5}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            hook = FinalModelEvalHook(
                eval_args={
                    "eval_data_path": eval_path,
                    "task_type": "scienceqa",
                    "max_samples": 100,
                    "final_eval_full": True,
                },
                output_dir=tmpdir,
            )
            hook.on_train_start({"model": model, "tokenizer": object()})
            final_metrics = {
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "invalid_rate": 0.0,
                "num_samples": 250,
            }
            with mock.patch.object(
                baseline_runner, "run_shared_eval", return_value=final_metrics
            ) as run_eval:
                hook.on_train_end({})

            self.assertIsNone(run_eval.call_args.kwargs["max_samples"])
            summary = json.loads(
                Path(tmpdir, "final_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["best_round"], 2)
            self.assertEqual(summary["best_accuracy"], 0.8)
            self.assertEqual(summary["best_macro_f1"], 0.6)
            self.assertEqual(summary["best_macro_f1_round"], 1)

    def test_missing_eval_path_disables_round_eval_with_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = PerRoundModelEvalHook(
                eval_args={
                    "eval_data_path": os.path.join(tmpdir, "missing.jsonl"),
                    "eval_freq": 1,
                    "max_samples": 100,
                },
                output_dir=tmpdir,
            )
            with mock.patch("builtins.print") as print_mock:
                hook.on_train_start({"model": _FakeModel(), "tokenizer": object()})
                hook.on_aggregate_end({"round_idx": 0})

            self.assertTrue(any("disabled" in str(call) for call in print_mock.call_args_list))
            self.assertFalse(Path(tmpdir, "eval_metrics_per_round.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
