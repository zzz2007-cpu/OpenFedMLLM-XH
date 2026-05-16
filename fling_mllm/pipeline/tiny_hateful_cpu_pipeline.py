"""CPU-only Tiny Hateful Memes FedAvg smoke runner.

This path is intentionally separate from MiniCPM-V. It exercises the real
Hateful Memes loader, client split resolver, FedAvg aggregation, evaluation,
checkpoint, and log writing without loading a large MLLM.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..federated.aggregation import get_clients_this_round, global_aggregate
from ..tasks import build_task_loader_kwargs, load_task_samples, resolve_client_data_path


class TinyHatefulMemesClassifier(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.encoder = nn.Linear(self.vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, bow: torch.Tensor, labels: torch.Tensor | None = None):
        hidden = torch.relu(self.encoder(bow.float()))
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels.long())
        return {"loss": loss, "logits": logits}


def _tokenize_to_bow(text: str, vocab_size: int) -> torch.Tensor:
    vec = torch.zeros(vocab_size, dtype=torch.float32)
    for token in str(text or "").lower().split():
        bucket = hash(token) % vocab_size
        vec[bucket] += 1.0
    denom = vec.sum().clamp_min(1.0)
    return vec / denom


class TinyHatefulDataset(Dataset):
    def __init__(self, samples: List[Dict], vocab_size: int):
        self.samples = samples
        self.vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        label = int(sample["label"])
        if label not in {0, 1}:
            raise ValueError(f"Invalid label for sample {sample.get('id')}: {label}")
        image_path = sample.get("image")
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path missing for sample {sample.get('id')}: {image_path}")
        return {
            "bow": _tokenize_to_bow(sample.get("text", ""), self.vocab_size),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def _plain_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        return asdict(obj)
    except Exception:
        try:
            return dict(obj)
        except Exception:
            return {}


def _state_stats(state: Dict[str, torch.Tensor], key: str) -> Dict[str, float]:
    tensor = state[key].float()
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
    }


def _state_delta_l2(after: Dict[str, torch.Tensor], before: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        diff = after[key].float() - before[key].float()
        total += float(torch.sum(diff * diff).item())
    return math.sqrt(total)


def _trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}


def _load_trainable_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    current = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            current[name].copy_(tensor.to(current[name].device))


def _label_dist(samples: Iterable[Dict]) -> Dict[str, int]:
    counter = Counter(int(sample["label"]) for sample in samples)
    return {str(label): int(counter.get(label, 0)) for label in (0, 1)}


def _load_client_samples(data_args, client_path: str) -> List[Dict]:
    loader_kwargs = build_task_loader_kwargs(task_type=data_args.task_type, data_args=data_args)
    loader_kwargs.setdefault("require_answer", True)
    samples = load_task_samples(
        data_path=client_path,
        task_type=data_args.task_type,
        split=getattr(data_args, "train_split", "train"),
        data_format=getattr(data_args, "data_format", "auto"),
        **loader_kwargs,
    )
    max_samples = getattr(data_args, "max_train_samples_per_client", None)
    if max_samples is not None and int(max_samples) > 0:
        samples = samples[: int(max_samples)]
    return samples


def _evaluate(model: nn.Module, samples: List[Dict], vocab_size: int, batch_size: int) -> Dict:
    dataset = TinyHatefulDataset(samples, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["bow"], labels=batch["labels"])
            losses.append(float(out["loss"].item()))
            pred = torch.argmax(out["logits"], dim=-1)
            preds.extend(int(x) for x in pred.tolist())
            labels.extend(int(x) for x in batch["labels"].tolist())
    total = len(labels)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    accuracy = correct / total if total else 0.0
    f1s = []
    for label in (0, 1):
        tp = sum(1 for p, y in zip(preds, labels) if p == label and y == label)
        fp = sum(1 for p, y in zip(preds, labels) if p == label and y != label)
        fn = sum(1 for p, y in zip(preds, labels) if p != label and y == label)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    cm = [[0, 0], [0, 0]]
    for p, y in zip(preds, labels):
        cm[y][p] += 1
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy),
        "f1_macro": float(sum(f1s) / len(f1s)),
        "num_samples": int(total),
        "confusion_matrix": cm,
        "label_names": ["no", "yes"],
    }


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_tiny_hateful_memes_fedavg_cpu(model_args, data_args, training_args, fed_args, eval_args=None):
    if getattr(model_args, "model_name_or_path", "") != "tiny_hateful_memes_cpu":
        raise ValueError("Tiny CPU runner only supports model_name_or_path='tiny_hateful_memes_cpu'.")

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tiny_fedavg_trace.jsonl"
    if log_path.exists():
        log_path.unlink()

    def trace(event: str, payload: dict) -> None:
        record = {"event": event, **payload}
        print("[TinyFedAvg] " + json.dumps(record, ensure_ascii=False), flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    torch.manual_seed(int(getattr(training_args, "seed", 42)))
    vocab_size = int(os.environ.get("OPENFED_TINY_VOCAB_SIZE", "512"))
    hidden_dim = int(os.environ.get("OPENFED_TINY_HIDDEN_DIM", "32"))
    model = TinyHatefulMemesClassifier(vocab_size=vocab_size, hidden_dim=hidden_dim)
    global_dict = _trainable_state_dict(model)
    local_dict_list = [dict((k, v.clone()) for k, v in global_dict.items()) for _ in range(fed_args.num_clients)]
    sample_num_list: List[int] = []
    local_samples: List[List[Dict]] = []

    for client_idx in range(fed_args.num_clients):
        path = resolve_client_data_path(data_args.data_path, client_idx)
        samples = _load_client_samples(data_args, path)
        local_samples.append(samples)
        sample_num_list.append(len(samples))

    eval_args = _plain_dict(eval_args)
    eval_path = eval_args.get("eval_data_path") or data_args.eval_data_path
    eval_loader_kwargs = build_task_loader_kwargs(task_type=data_args.task_type, data_args=data_args)
    eval_loader_kwargs["require_answer"] = True
    eval_samples = load_task_samples(
        data_path=eval_path,
        task_type=data_args.task_type,
        split=getattr(data_args, "eval_split", "eval"),
        data_format=getattr(data_args, "data_format", "auto"),
        **eval_loader_kwargs,
    )
    max_eval = eval_args.get("max_samples", None)
    if max_eval is not None and int(max_eval) > 0:
        eval_samples = eval_samples[: int(max_eval)]

    trainable_keys = list(global_dict.keys())
    watched_key = trainable_keys[0]
    trace("train_start", {
        "model": "tiny_hateful_memes_cpu",
        "trainable_keys": trainable_keys,
        "trainable_param_count": int(sum(v.numel() for v in global_dict.values())),
        "watched_key": watched_key,
        "num_clients": int(fed_args.num_clients),
        "sample_clients": int(fed_args.sample_clients),
        "client_sample_counts": sample_num_list,
        "client_label_distribution": {
            f"client_{i}": _label_dist(samples) for i, samples in enumerate(local_samples)
        },
        "eval_samples": len(eval_samples),
    })

    round_summaries = []
    batch_size = int(training_args.per_device_train_batch_size)
    local_epochs = int(training_args.num_train_epochs)
    max_steps = int(getattr(training_args, "max_steps", -1))

    for round_idx in range(int(fed_args.num_rounds)):
        clients_this_round = get_clients_this_round(fed_args, round_idx)
        trace("round_start", {
            "round_idx": round_idx,
            "clients_this_round": [int(c) for c in clients_this_round],
            "global_state_start": _state_stats(global_dict, watched_key),
        })
        local_losses = {}
        for client_idx in clients_this_round:
            _load_trainable_state_dict(model, global_dict)
            start_dict = _trainable_state_dict(model)
            start_delta = _state_delta_l2(start_dict, global_dict)
            dataset = TinyHatefulDataset(local_samples[client_idx], vocab_size=vocab_size)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            opt = torch.optim.SGD(model.parameters(), lr=float(training_args.learning_rate))
            model.train()
            losses = []
            steps = 0
            for _ in range(local_epochs):
                for batch in loader:
                    opt.zero_grad(set_to_none=True)
                    out = model(batch["bow"], labels=batch["labels"])
                    loss = out["loss"]
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                    steps += 1
                    if max_steps > 0 and steps >= max_steps:
                        break
                if max_steps > 0 and steps >= max_steps:
                    break
            local_state = _trainable_state_dict(model)
            local_dict_list[client_idx] = local_state
            local_losses[int(client_idx)] = float(np.mean(losses)) if losses else float("nan")
            trace("client_end", {
                "round_idx": round_idx,
                "client_idx": int(client_idx),
                "start_delta_from_global_l2": float(start_delta),
                "uploaded_keys": list(local_state.keys()),
                "uploaded_key_count": len(local_state),
                "loss": local_losses[int(client_idx)],
                "local_steps": steps,
                "local_delta_l2": _state_delta_l2(local_state, global_dict),
            })

        before_stats = _state_stats(global_dict, watched_key)
        global_dict, _ = global_aggregate(
            fed_args=fed_args,
            global_dict=global_dict,
            local_dict_list=local_dict_list,
            sample_num_list=sample_num_list,
            clients_this_round=clients_this_round,
            round_idx=round_idx,
        )
        after_stats = _state_stats(global_dict, watched_key)
        _load_trainable_state_dict(model, global_dict)
        metrics = _evaluate(model, eval_samples, vocab_size=vocab_size, batch_size=batch_size)
        round_record = {
            "round_idx": round_idx,
            "client_losses": local_losses,
            "aggregated_key_count": len(global_dict),
            "aggregated_keys": list(global_dict.keys()),
            "before_aggregate": before_stats,
            "after_aggregate": after_stats,
            "eval": metrics,
        }
        round_summaries.append(round_record)
        trace("aggregate_end", round_record)

        ckpt_dir = output_dir / f"checkpoint-{round_idx + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(global_dict, ckpt_dir / "tiny_global_state.pt")
        _save_json(ckpt_dir / "tiny_checkpoint_meta.json", round_record)

    summary = {
        "mode": "tiny_hateful_memes_cpu_fedavg",
        "data_path": data_args.data_path,
        "eval_data_path": eval_path,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "trainable_keys": trainable_keys,
        "sample_num_list": sample_num_list,
        "rounds": round_summaries,
    }
    _save_json(output_dir / "tiny_fedavg_summary.json", summary)
    _save_json(output_dir / "eval_metrics.json", round_summaries[-1]["eval"] if round_summaries else {})
    return summary
