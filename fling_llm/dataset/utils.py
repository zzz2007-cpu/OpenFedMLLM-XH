from typing import List

from torch.utils.data import Dataset


class CyclingDataset(Dataset):

    def __init__(self, data: Dataset):
        self.data = data
        self.start_idx = 0
        self.idxes = []

    def update(self, train_num: int):
        self.idxes = []
        for i in range(train_num + 1):
            self.idxes.append((i + self.start_idx) % len(self.data))
        self.start_idx = (self.start_idx + train_num) % len(self.data)
        print(f'now={self.start_idx}')

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        # print(self.idxes[idx])
        # if self.idxes[idx] == 1:
        #     print(self.data[self.idxes[idx]])
        if len(self.idxes) == 0:
            return self.data[idx]
        else:
            return self.data[self.idxes[idx]]


def get_union_dict_data(datasets: List[List]):
    # print(datasets)
    union_data = []
    for data in datasets:
        union_data += data
    return union_data


import copy
import json
import logging
import math
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"

_DATA_DEBUG_PRINTED = 0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _data_debug_enabled() -> bool:
    return _env_flag("OPENFED_DEBUG_DATA", default=False)


def _data_debug_limit(default: int = 3) -> int:
    raw = os.environ.get("OPENFED_DEBUG_DATA_LIMIT", str(default))
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _truncate_for_log(text, max_chars: int = 800) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def _last_role_text(conversations, role: str) -> str:
    for turn in reversed(conversations or []):
        if turn.get("role") == role:
            return str(turn.get("content", ""))
    return ""


def _decode_ids(tokenizer, token_ids) -> str:
    if token_ids is None or len(token_ids) == 0:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return "<decode_failed>"


def _debug_log_pre_tokenize(conversations, debug_meta):
    if not debug_meta or not debug_meta.get("enabled", False):
        return
    sample_idx = debug_meta.get("sample_idx", -1)
    raw_assistant = debug_meta.get("raw_assistant_text", "")
    raw_user = debug_meta.get("raw_user_text", "")
    prompt_text = _last_role_text(conversations, "user")
    target_text = _last_role_text(conversations, "assistant")
    print(
        "[DataDebug][PreTokenize] "
        f"sample_idx={sample_idx} "
        f"raw_user_repr={repr(_truncate_for_log(raw_user))} "
        f"raw_assistant_repr={repr(_truncate_for_log(raw_assistant))} "
        f"prompt_before_tokenizer_repr={repr(_truncate_for_log(prompt_text, 1200))} "
        f"target_after_preprocess_repr={repr(_truncate_for_log(target_text, 1200))}",
        flush=True,
    )


def _debug_log_label_supervision(ids, target, tokenizer, raw_msg, debug_meta):
    if not debug_meta or not debug_meta.get("enabled", False):
        return
    sample_idx = debug_meta.get("sample_idx", -1)
    non_ignore = target != -100
    supervised_ids = target[non_ignore].to(torch.long).tolist()
    supervised_text = _decode_ids(tokenizer, supervised_ids)
    full_input_text = _decode_ids(tokenizer, ids.to(torch.long).tolist())
    print(
        "[DataDebug][Labels] "
        f"sample_idx={sample_idx} "
        f"input_token_count={int(ids.numel())} "
        f"supervised_token_count={int(non_ignore.sum().item())} "
        f"labels_non_ignore_decode_repr={repr(_truncate_for_log(supervised_text, 1200))} "
        f"raw_msg_repr={repr(_truncate_for_log(raw_msg, 1200))} "
        f"full_input_decode_repr={repr(_truncate_for_log(full_input_text, 1200))}",
        flush=True,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            raw_data,
            transform,
            tokenizer,
            slice_config,
            llm_type="minicpm",
            patch_size=14,
            query_nums=64,
            batch_vision=False,
            max_length=2048,
            enable_audio=False,
            bad_sample_log_path=None,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums = query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length
        self.enable_audio = enable_audio
        self.bad_sample_log_path = bad_sample_log_path

    def __len__(self):
        return len(self.raw_data)

    def _build_sample(self, i):
        global _DATA_DEBUG_PRINTED
        debug_enabled = _data_debug_enabled()
        debug_limit = _data_debug_limit()
        do_debug = debug_enabled and (_DATA_DEBUG_PRINTED < debug_limit)
        raw_conversations = self.raw_data[i].get("conversations", [])
        debug_meta = None
        if do_debug:
            debug_meta = {
                "enabled": True,
                "sample_idx": i,
                "raw_user_text": _last_role_text(raw_conversations, "user"),
                "raw_assistant_text": _last_role_text(raw_conversations, "assistant"),
            }
        if isinstance(self.raw_data[i]["image"], str):
            images_dict = {"<image>": Image.open(self.raw_data[i]["image"]).convert("RGB")}
        elif isinstance(self.raw_data[i]["image"], Dict):
            images_dict = {img_name: Image.open(img_path).convert("RGB") for img_name, img_path in self.raw_data[i]["image"].items()}
        else:
            images_dict = None
        ret = preprocess(
            images_dict,
            self.raw_data[i]["conversations"],
            self.tokenizer,
            self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            llm_type=self.llm_type,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
            max_length=self.max_length,
            debug_meta=debug_meta,
        )
        if do_debug:
            _DATA_DEBUG_PRINTED += 1
        ret = dict(
            input_ids=ret["input_ids"],
            position_ids=ret["position_ids"],
            labels=ret["target"],
            attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            pixel_values=ret["pixel_values"],
            tgt_sizes=ret["tgt_sizes"],
            image_bound=ret["image_bound"],
        )
        if self.enable_audio and "audio" in self.raw_data[i]:
            audio_values = to_audio_tensor(self.raw_data[i]["audio"])
            ret["audio_values"] = audio_values
            ret["audio_attention_mask"] = torch.ones(audio_values.shape[0], dtype=torch.bool)
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        attempts = min(5, max(1, len(self.raw_data)))
        last_error = None
        current_idx = i
        for _ in range(attempts):
            try:
                return self._build_sample(current_idx)
            except Exception as e:
                last_error = e
                self._log_bad_sample(current_idx, e)
                current_idx = random.randint(0, len(self.raw_data) - 1)
        raise RuntimeError(f"Failed to build dataset sample after {attempts} retries: {last_error}")

    def _log_bad_sample(self, idx, err):
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


def data_collator(examples, padding_value=0, max_length=2048):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

    input_ids = trim_and_pad(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    if "position_ids" in examples[0]:
        position_ids = trim_and_pad(
            [example["position_ids"] for example in examples],
            batch_first=True,
            padding_value=0,
        )
    else:
        position_ids = None
    if "labels" in examples[0]:
        targets = trim_and_pad(
            [example["labels"] for example in examples],
            batch_first=True,
            padding_value=-100,
        )
    elif "target" in examples[0]:
        targets = trim_and_pad(
            [example["target"] for example in examples],
            batch_first=True,
            padding_value=-100,
        )
    else:
        targets = input_ids.clone()
        targets = targets.masked_fill(input_ids == padding_value, -100)
    if "attention_mask" in examples[0]:
        attention_mask = trim_and_pad(
            [example["attention_mask"] for example in examples],
            batch_first=True,
            padding_value=padding_value,
        )
    else:
        attention_mask = input_ids != padding_value
    pixel_values = [example["pixel_values"] for example in examples] if "pixel_values" in examples[0] else None
    image_bound = [example["image_bound"] for example in examples] if "image_bound" in examples[0] else None
    tgt_sizes = [example["tgt_sizes"] for example in examples] if "tgt_sizes" in examples[0] else None
    audio_values = [example["audio_values"] for example in examples] if "audio_values" in examples[0] else None
    audio_attention_mask = [example["audio_attention_mask"] for example in examples] if "audio_attention_mask" in examples[0] else None
    batch = {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_mask,
    }
    if position_ids is not None:
        batch["position_ids"] = position_ids
    if pixel_values is not None:
        batch["pixel_values"] = pixel_values
    if image_bound is not None:
        batch["image_bound"] = image_bound
    if tgt_sizes is not None:
        batch["tgt_sizes"] = tgt_sizes
    if audio_values is not None:
        batch["audio_values"] = pad_sequence(audio_values, batch_first=True, padding_value=0.0)
    if audio_attention_mask is not None:
        batch["audio_attention_mask"] = pad_sequence(audio_attention_mask, batch_first=True, padding_value=0)
    return batch


def to_audio_tensor(audio):
    if torch.is_tensor(audio):
        audio_tensor = audio.float()
    elif isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    elif isinstance(audio, list):
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
    elif isinstance(audio, str):
        if audio.endswith(".npy"):
            audio_tensor = torch.from_numpy(np.load(audio)).float()
        elif audio.endswith(".pt") or audio.endswith(".pth"):
            loaded = torch.load(audio, map_location="cpu")
            if torch.is_tensor(loaded):
                audio_tensor = loaded.float()
            else:
                audio_tensor = torch.tensor(loaded, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported audio format: {audio}")
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(-1)
    return audio_tensor


def conversation_to_ids(conversation, tokenizer, llm_type=None, new_schema=False, max_length=2048, in_type='image', debug_meta=None):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(
            conversation, tokenizer
        )
    elif llm_type == "qwen2":
        input_ids, context, raw_msg = conversation_to_ids_qwen2(
            conversation, tokenizer
        )
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(
            conversation, tokenizer
        )

    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))
    if ids.shape[-1] > max_length:
        ids = ids[:max_length]
        context = context[:max_length]
        logger.warning(
            f"The input length ({ids.shape[-1]}) exceeds the model's maximum length ({max_length}), so it has been truncated")

    if torch.all(context):
        if len(context) > 1:
            context[-1] = 0
            logger.warning("No assistant tokens found; fallback to last-token supervision.")
        else:
            logger.error("No tokens available to compute loss.")
            raise Exception("No tokens available to compute loss.")

    # build target in unshifted form:
    # labels align with input_ids positions and will be shifted in loss computation.
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(len(ids)):
        if context[i] == 0:
            target[i] = ids[i]
    # For assistant->non-assistant boundaries in multi-turn chat, keep an explicit
    # end-of-turn supervision token.
    for i in range(1, len(ids)):
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i] = tokenizer.eot_id
            else:
                target[i] = tokenizer.eos_id

    if in_type == 'image':
        # build image bound
        if new_schema:
            start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
            end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
            image_start_tokens = torch.where(start_cond)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]
        else:
            image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
        if len(image_start_tokens) > 0 and len(image_end_tokens) > 0:
            valid_pairs = []
            end_idx = 0
            for start_token in image_start_tokens.tolist():
                while end_idx < len(image_end_tokens) and image_end_tokens[end_idx].item() <= start_token:
                    end_idx += 1
                if end_idx >= len(image_end_tokens):
                    break
                valid_pairs.append([start_token, image_end_tokens[end_idx].item()])
                end_idx += 1
            if valid_pairs:
                image_bound = torch.tensor(valid_pairs, dtype=torch.long)
            else:
                image_bound = []
        else:
            image_bound = []
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()
    _debug_log_label_supervision(ids, target, tokenizer, raw_msg, debug_meta)
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }


def conversation_to_ids_minicpm(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "<用户>"
        else:
            prefix = "<AI>"
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        message_ids = tokenizer.encode(message, add_special_tokens=False)

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg


def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3  # assistant<|end_header_id|>\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg


def conversation_to_ids_qwen2(conversation, tokenizer):
    raw_msg = ""
    chat = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "user"
        else:
            prefix = "assistant"
        chat.append({"role": prefix, "content": message})
        raw_msg += prefix + message
    assert set([i['role'] for i in chat]) & set(['assistant'])

    ret = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids)

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_start|>'))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('assistant'))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx - 1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context, raw_msg


def preprocess(
        images_dict,
        conversations,
        tokenizer,
        transform,
        query_nums=64,
        slice_config=None,
        llm_type=None,
        patch_size=14,
        batch_vision=False,
        max_length=2048,
        debug_meta=None,
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 1, "conversations length must large than 2"
    assert conversations[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    if images_dict is not None:
        default_image_placeholder = (
                tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
        )
    new_schema = False
    use_image_id = False
    if llm_type == 'qwen2':
        new_schema = True
        use_image_id = True
    if images_dict is not None:
        image_placeholder_dict = {}
        images = []
        image_id_cnt = 0
        for img_name, image in images_dict.items():
            if slice_config:
                source_image, patches, best_grid = slice_image(
                    image,
                    slice_config["max_slice_nums"],
                    slice_config["scale_resolution"],
                    slice_config["patch_size"],
                )
                images.append(source_image)
                image_placeholder = default_image_placeholder
                if len(patches) > 0:
                    for i in range(len(patches)):
                        for j in range(len(patches[0])):
                            images.append(patches[i][j])
                    if use_image_id:
                        image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                        image_id_cnt += 1
                    image_placeholder += get_grid_placeholder(
                        tokenizer, best_grid, query_nums, new_schema=new_schema)
                image_placeholder_dict[img_name] = image_placeholder
            else:
                images.append(image)
                image_placeholder = default_image_placeholder
                if use_image_id:
                    image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                    image_id_cnt += 1
                image_placeholder_dict[img_name] = image_placeholder

        images = [transform(i) for i in images]

        if len(images_dict) == 1 and "<image>" in images_dict:
            single_image_placeholder = image_placeholder_dict["<image>"]
            # print("images_dict", images_dict)
            if "<image>" in conversations[0]["content"]:
                conversations[0]["content"] = conversations[0]["content"].replace(
                    "<image>", single_image_placeholder
                )
            else:
                conversations[0]["content"] = (
                        single_image_placeholder + "\n" + conversations[0]["content"]
                )
            _debug_log_pre_tokenize(conversations, debug_meta)
            input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length, debug_meta=debug_meta)
        else:
            pattern = r'<image_\d+>'
            new_conversations = []
            for conversation in conversations:
                content = conversation['content']
                parts = re.split(f'({pattern})', content)
                for i, part in enumerate(parts):
                    if not part.strip():
                        continue
                    if re.match(pattern, part):
                        if part in image_placeholder_dict:
                            parts[i] = image_placeholder_dict[part]
                        else:
                            raise Exception(f"not found {part} in image dict")
                conversation['content'] = '\n'.join(parts)
                new_conversations.append(conversation)
            conversations = new_conversations

            _debug_log_pre_tokenize(conversations, debug_meta)
            input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length, debug_meta=debug_meta)

        if batch_vision:
            tgt_sizes = []
            reshape_images = []
            for image in images:
                H, W = image.shape[1:]
                reshape_image = reshape_by_patch(image, patch_size)
                reshape_images.append(reshape_image)
                tgt_sizes.append([H // patch_size, W // patch_size])
            if tgt_sizes:
                tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)
            image_bound = input_dict["image_bound"]
            if torch.is_tensor(image_bound):
                keep_n = image_bound.size(0)
                reshape_images = reshape_images[:keep_n]
                if torch.is_tensor(tgt_sizes):
                    tgt_sizes = tgt_sizes[:keep_n]

            input_dict["pixel_values"] = reshape_images
            input_dict["tgt_sizes"] = tgt_sizes

        else:
            image_bound = input_dict["image_bound"]
            if torch.is_tensor(image_bound):
                images = images[: image_bound.size(0)]
            input_dict["pixel_values"] = images
            input_dict["tgt_sizes"] = []
    else:
        _debug_log_pre_tokenize(conversations, debug_meta)
        input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length, 'text', debug_meta=debug_meta)
        input_dict["pixel_values"] = []
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(
        image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
            (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
        original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = (
                tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
        )
    else:
        image_placeholder = (
                tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
        )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    if new_schema:
        slice_placeholder = '\n'.join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + \
                            "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches
