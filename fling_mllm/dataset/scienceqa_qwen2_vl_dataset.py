"""Qwen2-VL supervised dataset for ScienceQA option-letter training."""

from __future__ import annotations

import json
import os
import random
from functools import partial
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..adapters.conversation import build_qwen2_vl_messages
from ..tasks.scienceqa.prompt_builder import answer_letter_from_sample, build_scienceqa_prompt, image_path_for_context


def _move_if_tensor(value):
    return value.squeeze(0) if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == 1 else value


class ScienceQAQwen2VLDataset(Dataset):
    def __init__(
        self,
        raw_data: List[Dict],
        processor,
        tokenizer,
        max_length: int = 2048,
        bad_sample_log_path: Optional[str] = None,
    ):
        self.raw_data = raw_data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.bad_sample_log_path = bad_sample_log_path

    def __len__(self):
        return len(self.raw_data)

    def _load_image(self, image_path: Optional[str]):
        if not image_path:
            return None
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _encode_view(self, prompt: str, target: str, image_path: Optional[str]) -> Dict[str, torch.Tensor]:
        image = self._load_image(image_path)
        full_messages = build_qwen2_vl_messages(
            user_text=prompt,
            image_path=image_path if image is not None else None,
            assistant_text=target,
        )
        prompt_messages = build_qwen2_vl_messages(
            user_text=prompt,
            image_path=image_path if image is not None else None,
            assistant_text=None,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_kwargs = dict(return_tensors="pt", padding=False)
        if image is not None:
            full_inputs = self.processor(text=[full_text], images=[image], **processor_kwargs)
            prompt_inputs = self.processor(text=[prompt_text], images=[image], **processor_kwargs)
        else:
            full_inputs = self.processor(text=[full_text], **processor_kwargs)
            prompt_inputs = self.processor(text=[prompt_text], **processor_kwargs)

        input_ids = full_inputs["input_ids"].squeeze(0).long()
        attention_mask = full_inputs["attention_mask"].squeeze(0).long()
        labels = input_ids.clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        labels[:prompt_len] = -100

        if input_ids.size(0) > self.max_length:
            start = input_ids.size(0) - self.max_length
            input_ids = input_ids[start:]
            attention_mask = attention_mask[start:]
            labels = labels[start:]

        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key, value in full_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            ret[key] = _move_if_tensor(value)
        return ret

    def _build_masked_view(self, sample: dict, idx: int, target: str):
        ctx = sample.get("available_context") or {}
        if not (ctx.get("image") and ctx.get("text_context")):
            return None
        # Deterministic alternation keeps the CPU smoke reproducible while
        # exercising both modality masks over a dataset.
        if idx % 2 == 0:
            masked_prompt = build_scienceqa_prompt(sample, include_text_context=False)
            masked_image = image_path_for_context(sample)
        else:
            masked_prompt = sample.get("prompt") or build_scienceqa_prompt(sample)
            masked_image = None
        return self._encode_view(masked_prompt, target, masked_image)

    def _build_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_data[idx]
        prompt = sample.get("prompt") or build_scienceqa_prompt(sample)
        target = answer_letter_from_sample(sample)
        image_path = image_path_for_context(sample)
        encoded = self._encode_view(prompt, target, image_path)
        ctx = sample.get("available_context") or {}
        ret = {
            **encoded,
            "scienceqa_full_context": torch.tensor(
                int(bool(ctx.get("image")) and bool(ctx.get("text_context"))),
                dtype=torch.long,
            ),
            "scienceqa_has_image": torch.tensor(int(bool(ctx.get("image"))), dtype=torch.long),
            "scienceqa_has_text_context": torch.tensor(int(bool(ctx.get("text_context"))), dtype=torch.long),
        }
        masked = self._build_masked_view(sample, idx=idx, target=target)
        if masked is not None:
            for key, value in masked.items():
                ret[f"scienceqa_masked_{key}"] = value
        return ret

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        attempts = min(5, max(1, len(self.raw_data)))
        cur_idx = idx
        last_error = None
        for _ in range(attempts):
            try:
                return self._build_sample(cur_idx)
            except Exception as exc:
                last_error = exc
                self._log_bad_sample(cur_idx, exc)
                cur_idx = random.randint(0, len(self.raw_data) - 1)
        raise RuntimeError(f"Failed to build ScienceQA Qwen2-VL sample after {attempts} retries: {last_error}")

    def _log_bad_sample(self, idx: int, err: Exception):
        if not self.bad_sample_log_path:
            return
        sample = self.raw_data[idx] if 0 <= idx < len(self.raw_data) else {}
        record = {
            "idx": idx,
            "error": str(err),
            "id": sample.get("id"),
            "image": sample.get("image"),
        }
        log_dir = os.path.dirname(self.bad_sample_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.bad_sample_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pad_1d(sequences, padding_value, max_length):
    trimmed = [seq[:max_length] for seq in sequences]
    return pad_sequence(trimmed, batch_first=True, padding_value=padding_value)


def scienceqa_qwen2_vl_data_collator(examples, pad_token_id=0, max_length=2048):
    batch = {
        "input_ids": _pad_1d([x["input_ids"] for x in examples], pad_token_id, max_length),
        "attention_mask": _pad_1d([x["attention_mask"] for x in examples], 0, max_length),
        "labels": _pad_1d([x["labels"] for x in examples], -100, max_length),
        "scienceqa_full_context": torch.stack([x["scienceqa_full_context"] for x in examples]),
        "scienceqa_has_image": torch.stack([x["scienceqa_has_image"] for x in examples]),
        "scienceqa_has_text_context": torch.stack([x["scienceqa_has_text_context"] for x in examples]),
    }
    if all("scienceqa_masked_input_ids" in x for x in examples):
        batch["scienceqa_masked_input_ids"] = _pad_1d(
            [x["scienceqa_masked_input_ids"] for x in examples], pad_token_id, max_length
        )
        batch["scienceqa_masked_attention_mask"] = _pad_1d(
            [x["scienceqa_masked_attention_mask"] for x in examples], 0, max_length
        )
        batch["scienceqa_masked_labels"] = _pad_1d(
            [x["scienceqa_masked_labels"] for x in examples], -100, max_length
        )
        if "scienceqa_masked_pixel_values" in examples[0]:
            values = [x["scienceqa_masked_pixel_values"] for x in examples]
            if all(torch.is_tensor(v) for v in values):
                try:
                    batch["scienceqa_masked_pixel_values"] = torch.cat(values, dim=0)
                except Exception:
                    batch["scienceqa_masked_pixel_values"] = values
        if "scienceqa_masked_image_grid_thw" in examples[0]:
            grids = [x["scienceqa_masked_image_grid_thw"] for x in examples]
            if all(torch.is_tensor(g) for g in grids):
                try:
                    batch["scienceqa_masked_image_grid_thw"] = torch.cat(grids, dim=0)
                except Exception:
                    batch["scienceqa_masked_image_grid_thw"] = grids
    if "pixel_values" in examples[0]:
        values = [x["pixel_values"] for x in examples]
        if all(torch.is_tensor(v) for v in values):
            try:
                batch["pixel_values"] = torch.cat(values, dim=0)
            except Exception:
                batch["pixel_values"] = values
    if "image_grid_thw" in examples[0]:
        grids = [x["image_grid_thw"] for x in examples]
        if all(torch.is_tensor(g) for g in grids):
            try:
                batch["image_grid_thw"] = torch.cat(grids, dim=0)
            except Exception:
                batch["image_grid_thw"] = grids
    return batch


def build_scienceqa_qwen2_vl_data_module(
    train_json,
    tokenizer,
    processor,
    max_length=2048,
    bad_sample_log_path=None,
):
    train_dataset = ScienceQAQwen2VLDataset(
        raw_data=train_json,
        processor=processor,
        tokenizer=tokenizer,
        max_length=max_length,
        bad_sample_log_path=bad_sample_log_path,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": partial(
            scienceqa_qwen2_vl_data_collator,
            pad_token_id=pad_token_id,
            max_length=max_length,
        ),
    }
