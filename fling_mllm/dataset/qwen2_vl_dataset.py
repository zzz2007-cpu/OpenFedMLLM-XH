import json
import os
import random
from functools import partial
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..adapters.conversation import build_qwen2_vl_messages, extract_user_assistant_turns


def _move_if_tensor(value):
    return value.squeeze(0) if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == 1 else value


class Qwen2VLSupervisedDataset(Dataset):
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
        if image_path is None:
            return None
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _build_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_data[idx]
        image_path = sample.get("image")
        conversations = sample.get("conversations", [])
        user_text, assistant_text = extract_user_assistant_turns(conversations)

        if not assistant_text:
            raise ValueError("Missing assistant target text in conversations.")

        image = self._load_image(image_path) if isinstance(image_path, str) else None

        full_messages = build_qwen2_vl_messages(
            user_text=user_text,
            image_path=image_path if image is not None else None,
            assistant_text=assistant_text,
        )
        prompt_messages = build_qwen2_vl_messages(
            user_text=user_text,
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
            # Keep the latest tokens to preserve assistant supervision tail.
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
        raise RuntimeError(f"Failed to build qwen2-vl sample after {attempts} retries: {last_error}")

    def _log_bad_sample(self, idx: int, err: Exception):
        if not self.bad_sample_log_path:
            return
        sample = self.raw_data[idx] if 0 <= idx < len(self.raw_data) else {}
        record = {
            "idx": idx,
            "error": str(err),
            "image": sample.get("image"),
            "conversations": sample.get("conversations"),
        }
        log_dir = os.path.dirname(self.bad_sample_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.bad_sample_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pad_1d(sequences, padding_value, max_length):
    trimmed = [seq[:max_length] for seq in sequences]
    return pad_sequence(trimmed, batch_first=True, padding_value=padding_value)


def qwen2_vl_data_collator(examples, pad_token_id=0, max_length=2048):
    input_ids = _pad_1d([x["input_ids"] for x in examples], pad_token_id, max_length)
    attention_mask = _pad_1d([x["attention_mask"] for x in examples], 0, max_length)
    labels = _pad_1d([x["labels"] for x in examples], -100, max_length)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Qwen2-VL vision features are variable-length across samples.
    if "pixel_values" in examples[0]:
        pixel_values = [x["pixel_values"] for x in examples]
        if all(torch.is_tensor(v) for v in pixel_values):
            try:
                batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            except Exception:
                batch["pixel_values"] = pixel_values
        else:
            batch["pixel_values"] = pixel_values

    if "image_grid_thw" in examples[0]:
        grids = [x["image_grid_thw"] for x in examples]
        if all(torch.is_tensor(g) for g in grids):
            try:
                batch["image_grid_thw"] = torch.cat(grids, dim=0)
            except Exception:
                batch["image_grid_thw"] = grids
        else:
            batch["image_grid_thw"] = grids

    if "pixel_attention_mask" in examples[0]:
        masks = [x["pixel_attention_mask"] for x in examples]
        if all(torch.is_tensor(m) for m in masks):
            try:
                batch["pixel_attention_mask"] = torch.cat(masks, dim=0)
            except Exception:
                batch["pixel_attention_mask"] = masks
        else:
            batch["pixel_attention_mask"] = masks
    return batch


def build_qwen2_vl_data_module(
    train_json,
    tokenizer,
    processor,
    max_length=2048,
    bad_sample_log_path=None,
):
    train_dataset = Qwen2VLSupervisedDataset(
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
            qwen2_vl_data_collator,
            pad_token_id=pad_token_id,
            max_length=max_length,
        ),
    }
