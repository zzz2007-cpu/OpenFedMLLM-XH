"""
Evaluation utilities for federated multimodal LLM benchmark.

Provides:
- load_model_for_eval(): load merged LoRA model from checkpoint
- generate_answer():     single-sample multimodal inference
- compute_classification_metrics(): Accuracy / F1 / per-class report
- correct_image_paths(): same path-correction logic used in training
"""

import os
import json
import re
import warnings
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from transformers import AutoModel, AutoTokenizer, AutoProcessor
try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover - compatibility for older transformers
    AutoModelForVision2Seq = None
from ..adapters import (
    MODEL_FAMILY_QWEN2_VL,
    infer_model_family_from_model,
    resolve_model_family,
)
from ..adapters.conversation import build_qwen2_vl_messages, strip_image_placeholders


# ---------------------------------------------------------------------------
# Image path correction (mirrors fling_mllm/dataset/dataset.py)
# ---------------------------------------------------------------------------

def correct_image_paths(samples: List[Dict], project_root: str) -> List[Dict]:
    """Fix relative/broken image paths in the test JSON, matching training logic."""
    real_image_root = os.path.join(
        project_root, "data", "crisis-mmd", "raw_data", "data_image"
    )
    for item in samples:
        if "image" in item and isinstance(item["image"], str):
            if "data_image" in item["image"]:
                parts = item["image"].split("data_image")
                if len(parts) > 1:
                    rel_path = parts[-1].lstrip("/\\")
                    item["image"] = os.path.join(real_image_root, rel_path)
    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_eval(
    checkpoint_dir: str,
    base_model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "auto",
) -> Tuple:
    """
    Load a PEFT/LoRA adapter merged on top of the base model.

    Args:
        checkpoint_dir: Path that contains adapter_config.json + adapter weights
                        (e.g. ./mllmzoo/output/quick_run/checkpoint-2)
        base_model_name: HuggingFace model id (e.g. openbmb/MiniCPM-V-2_6-int4)
        cache_dir:       Optional cache directory.
        device:          'auto', 'cuda', or 'cpu'
    Returns:
        (model, tokenizer) ready for inference.
    """
    model_family = resolve_model_family(base_model_name)
    # Detect if checkpoint has PEFT adapter config
    has_adapter = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))

    # Check bitsandbytes availability (required for int4 base)
    requires_bnb = "int4" in base_model_name.lower() or "int8" in base_model_name.lower()
    if requires_bnb and importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "bitsandbytes is required for int4/int8 model evaluation. "
            "Install with: pip install bitsandbytes"
        )

    compute_dtype = torch.bfloat16 if (device != "cpu") else torch.float32
    if device == "auto":
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    else:
        device_map = device
    trust_remote_code = True

    if model_family == MODEL_FAMILY_QWEN2_VL:
        if AutoModelForVision2Seq is None:
            raise ImportError(
                "Qwen2-VL evaluation requires transformers with AutoModelForVision2Seq."
            )
        model_loader = AutoModelForVision2Seq
    else:
        model_loader = AutoModel

    if has_adapter:
        from peft import PeftModel
        base_model = model_loader.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=compute_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        # Merge LoRA weights into the base model for faster inference
        try:
            model = model.merge_and_unload()
        except Exception:
            # Some quantized models don't support merge_and_unload; use as-is
            pass
    else:
        # Plain checkpoint (e.g. full model saved with save_pretrained)
        model = model_loader.from_pretrained(
            checkpoint_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=compute_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    if model_family == MODEL_FAMILY_QWEN2_VL:
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        try:
            model.processor = processor
        except Exception:
            pass
    model._openfed_model_family = model_family
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _get_slice_config(model):
    """Return the slice_config object from a MiniCPM-V model, or None."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "slice_config"):
        return cfg.slice_config
    return None


def _resolve_max_slice_nums(model, explicit: Optional[int] = None) -> Optional[int]:
    if explicit is not None:
        return int(explicit)
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "slice_config") and hasattr(cfg.slice_config, "max_slice_nums"):
        return int(cfg.slice_config.max_slice_nums)
    if hasattr(cfg, "max_slice_nums"):
        return int(cfg.max_slice_nums)
    return None


def _get_chat_processor(model, max_slice_nums: Optional[int] = None):
    processor = getattr(model, "processor", None)
    if processor is None:
        model_name = getattr(getattr(model, "config", None), "_name_or_path", None)
        if not model_name:
            return None
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        try:
            model.processor = processor
        except Exception:
            pass
    image_processor = getattr(processor, "image_processor", None)
    if (
        max_slice_nums is not None
        and image_processor is not None
        and hasattr(image_processor, "max_slice_nums")
    ):
        image_processor.max_slice_nums = int(max_slice_nums)
    return processor


def _sync_chat_runtime_config(model, max_slice_nums: Optional[int]) -> None:
    """
    Keep chat-time runtime config coherent and avoid internal chunk-cat mismatch
    when one image is sliced into multiple tiles.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or max_slice_nums is None:
        return
    m = int(max_slice_nums)
    if hasattr(cfg, "slice_config") and hasattr(cfg.slice_config, "max_slice_nums"):
        cfg.slice_config.max_slice_nums = m
    elif hasattr(cfg, "max_slice_nums"):
        cfg.max_slice_nums = m
    current = int(getattr(cfg, "vision_batch_size", 1))
    cfg.vision_batch_size = max(current, m + 4)


_OPTION_LINE_RE = re.compile(r"^\s*\(([A-Z])\)\s*(.+?)\s*$")
_ANSWER_CUE_RE = re.compile(
    r"(?:\n)?(?:the best answer is:|the answer is:|answer:)\s*$",
    re.IGNORECASE,
)
_ANSWER_LETTER_RE = re.compile(
    r"(?:the answer is|answer)\s*[:：]?\s*\(?\s*([A-Z])\s*\)?",
    re.IGNORECASE,
)
_LEADING_LETTER_RE = re.compile(
    r"^\s*\(?\s*([A-Z])\s*\)?(?:[\.\):\-]|$|\s)",
    re.IGNORECASE,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _truncate_for_log(text: str, max_chars: int = 1000) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def _safe_decode(tokenizer, token_ids) -> str:
    if token_ids is None:
        return ""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        return f"<decode_failed: {type(exc).__name__}: {exc}>"


def _safe_encode(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        try:
            ret = tokenizer(text, add_special_tokens=False)
            ids = ret.get("input_ids", []) if isinstance(ret, dict) else []
            if isinstance(ids, list):
                return ids
        except Exception:
            pass
    return []


def _summarize_tensor_like(x: Any):
    if torch.is_tensor(x):
        return tuple(x.shape)
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    if isinstance(x, list):
        return f"list(len={len(x)})"
    return type(x).__name__


def _normalize_question_for_chat(question: str) -> str:
    """
    Remove explicit image placeholders from dataset text.
    MiniCPM chat already receives image content via `image=` or inline `[image, text]`.
    Keeping both causes placeholder/image-count mismatch at runtime.
    """
    return strip_image_placeholders(question)


def parse_question_options(question: Optional[str]) -> Dict[str, str]:
    """Extract the option table from a multiple-choice question."""
    options: Dict[str, str] = {}
    for line in (question or "").splitlines():
        m = _OPTION_LINE_RE.match(line)
        if m:
            options[m.group(1)] = m.group(2).strip()
    return options


def _constrain_question_to_letter_answer(question: str) -> str:
    """
    Strengthen the answer instruction for evaluation-time generation.

    The training/eval prompt already includes the option table, but many base
    VLMs still expand the chosen letter into the full option text. Appending a
    short explicit constraint and trimming the old answer cue makes generation
    much more likely to stay in the single-letter regime expected by the
    dataset.
    """
    text = _normalize_question_for_chat(question)
    options = parse_question_options(text)
    if not options:
        return text

    text = _ANSWER_CUE_RE.sub("", text).rstrip()
    letters = ", ".join(sorted(options))
    return (
        f"{text}\n\n"
        f"Respond with ONLY one uppercase option letter from {{{letters}}}.\n"
        "Do not output the option text, explanation, or punctuation.\n"
        "Answer:"
    )


def _infer_runtime_device(model, preferred_device: str = "cuda"):
    if preferred_device == "cpu":
        return torch.device("cpu")
    if preferred_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_qwen2_vl_inputs(
    model,
    question: str,
    image_path: Optional[str],
    device: str = "cuda",
):
    processor = _get_chat_processor(model, None)
    if processor is None:
        raise RuntimeError("Qwen2-VL processor is unavailable on model.")

    has_image = image_path is not None and os.path.exists(image_path)
    messages = build_qwen2_vl_messages(
        user_text=question,
        image_path=image_path if has_image else None,
        assistant_text=None,
    )
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    proc_kwargs = dict(return_tensors="pt", padding=True)
    if has_image:
        with Image.open(image_path) as _img:
            image = _img.convert("RGB")
        inputs = processor(text=[text_prompt], images=[image], **proc_kwargs)
    else:
        inputs = processor(text=[text_prompt], **proc_kwargs)

    runtime_device = _infer_runtime_device(model, preferred_device=device)
    inputs = {
        k: (v.to(runtime_device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }
    prompt_len = int(inputs["input_ids"].shape[-1])
    return inputs, prompt_len


def _generate_answer_qwen2_vl(
    model,
    tokenizer,
    question: str,
    image_path: Optional[str],
    max_new_tokens: int,
    device: str,
):
    inputs, prompt_len = _prepare_qwen2_vl_inputs(
        model=model,
        question=question,
        image_path=image_path,
        device=device,
    )
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    completion_ids = generated_ids[:, prompt_len:]
    processor = _get_chat_processor(model, None)
    if processor is not None and hasattr(processor, "batch_decode"):
        text = processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
    else:
        text = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
    return text.strip()


def generate_answer(
    model,
    tokenizer,
    question: str,
    image_path: Optional[str] = None,
    max_new_tokens: int = 32,
    max_slice_nums: Optional[int] = None,
    device: str = "cuda",
    enforce_letter_output: bool = False,
    debug_generation: Optional[bool] = None,
) -> str:
    """
    Run single-sample multimodal inference.

    Slicing is configuration-driven. By default this function uses
    `model.config.slice_config.max_slice_nums` (or `model.config.max_slice_nums`)
    and synchronizes the chat processor to the same value.

    Args:
        model: Loaded model (eval mode).
        tokenizer: Matching tokenizer.
        question: The user-side question text (may include '<image>' placeholder).
        image_path: Absolute path to the image; None for text-only.
        max_new_tokens: Max tokens to generate per sample.
        max_slice_nums: Optional override. If None, use model config.
        device: 'cuda' or 'cpu'.
    Returns:
        Decoded answer string (stripped).
    """
    # 单样本逐条推理，不修改 max_slice_nums。
    # model.config 和 image_processor 各自保持 config.json /
    # preprocessor_config.json 的原始值：两者天然一致，不会触发 ValueError。
    # （强制设为 1 会降低图像理解质量，且批量推理才需要统一切片数量。）
    debug_mode = _env_flag("OPENFED_DEBUG_EVAL", default=False) if debug_generation is None else bool(debug_generation)
    normalized_question = _normalize_question_for_chat(question)
    parsed_options = parse_question_options(normalized_question)
    if enforce_letter_output:
        normalized_question = _constrain_question_to_letter_answer(question)
        parsed_options = parse_question_options(normalized_question)
    effective_max_new_tokens = max_new_tokens
    if enforce_letter_output and parsed_options:
        # Leave a small buffer for tokenizers that emit whitespace/newline with
        # the letter, while strongly discouraging full option-text completions.
        effective_max_new_tokens = min(max_new_tokens, 4)
    if debug_mode:
        print(
            "[EvalDebug][GenerateInput] "
            f"enforce_letter_output={enforce_letter_output} "
            f"requested_max_new_tokens={max_new_tokens} "
            f"effective_max_new_tokens={effective_max_new_tokens} "
            f"has_image={bool(image_path)} "
            f"question_repr={repr(_truncate_for_log(normalized_question, 1200))}",
            flush=True,
        )
    if infer_model_family_from_model(model) == MODEL_FAMILY_QWEN2_VL:
        output_text = _generate_answer_qwen2_vl(
            model=model,
            tokenizer=tokenizer,
            question=normalized_question,
            image_path=image_path,
            max_new_tokens=effective_max_new_tokens,
            device=device,
        )
        if debug_mode:
            print(
                "[EvalDebug][GenerateOutput] "
                f"output_repr={repr(_truncate_for_log(output_text, 1200))}",
                flush=True,
            )
        return output_text
    resolved_max_slice_nums = _resolve_max_slice_nums(model, max_slice_nums)
    _sync_chat_runtime_config(model, resolved_max_slice_nums)
    processor = _get_chat_processor(model, resolved_max_slice_nums)
    # ------------------------------------------------------------------
    # MiniCPM-V logs "These two values should be the same. Check
    # config.json and preprocessor_config.json." via the transformers
    # logger every time it runs the image processor.  It is purely
    # informational and does NOT affect inference.  Suppress it here so
    # it doesn't spam the console.
    # ------------------------------------------------------------------
    import logging as _logging
    _transformers_logger = _logging.getLogger("transformers")
    _prev_level = _transformers_logger.level
    _transformers_logger.setLevel(_logging.ERROR)   # hide WARNING + INFO
    try:
        def _chat_with_fallback(base_kwargs: Dict[str, Any]):
            kwargs = dict(base_kwargs)
            if resolved_max_slice_nums is not None:
                kwargs["max_slice_nums"] = resolved_max_slice_nums
            if processor is not None:
                kwargs["processor"] = processor
            try:
                return model.chat(sampling=False, **kwargs)
            except TypeError:
                pass
            try:
                return model.chat(**kwargs)
            except TypeError:
                kwargs.pop("max_slice_nums", None)
                kwargs.pop("processor", None)
                try:
                    return model.chat(sampling=False, **kwargs)
                except TypeError:
                    return model.chat(**kwargs)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*These two values.*")
            has_image = image_path is not None and os.path.exists(image_path)
            response = None
            last_error = None
            if has_image:
                with Image.open(image_path) as _img:
                    image = _img.convert("RGB")
                chat_variants = [
                    # Preferred MiniCPM format: inline multimodal content.
                    {
                        "image": None,
                        "msgs": [{"role": "user", "content": [image, normalized_question]}],
                        "tokenizer": tokenizer,
                        "max_new_tokens": effective_max_new_tokens,
                    },
                    # Compatibility fallback for APIs that expect `image=...`.
                    {
                        "image": image,
                        "msgs": [{"role": "user", "content": normalized_question}],
                        "tokenizer": tokenizer,
                        "max_new_tokens": effective_max_new_tokens,
                    },
                ]
            else:
                chat_variants = [
                    {
                        "msgs": [{"role": "user", "content": normalized_question}],
                        "tokenizer": tokenizer,
                        "max_new_tokens": effective_max_new_tokens,
                    }
                ]

            for chat_kwargs in chat_variants:
                try:
                    response = _chat_with_fallback(chat_kwargs)
                    break
                except Exception as e:
                    last_error = e
                    continue
            if response is None and last_error is not None:
                raise last_error
        if isinstance(response, tuple):
            response = response[0]
        output_text = str(response)
        if debug_mode:
            print(
                "[EvalDebug][GenerateOutput] "
                f"output_repr={repr(_truncate_for_log(output_text, 1200))} "
                f"stripped_repr={repr(_truncate_for_log(output_text.strip(), 1200))}",
                flush=True,
            )
        return output_text.strip()
    finally:
        _transformers_logger.setLevel(_prev_level)


# ---------------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------------

def extract_question(sample: Dict) -> str:
    """Return the user-side content from conversations[0]."""
    convs = sample.get("conversations", [])
    for turn in convs:
        if turn.get("role") == "user":
            return turn.get("content", "")
    return ""


def extract_ground_truth(sample: Dict) -> str:
    """Return the last assistant-turn content as the ground-truth label."""
    convs = sample.get("conversations", [])
    for turn in reversed(convs):
        if turn.get("role") == "assistant":
            return turn.get("content", "").strip()
    return ""


_EVAL_SAMPLE_SNAPSHOT_PRINTED = False


def log_first_eval_sample_snapshot(
    sample: Dict,
    tokenizer,
    model=None,
    stage_tag: str = "Eval",
) -> None:
    """
    Print a one-time structured snapshot of the first evaluation sample:
    raw json / preprocessed question / tokenizer+processor outputs.
    """
    global _EVAL_SAMPLE_SNAPSHOT_PRINTED
    if _EVAL_SAMPLE_SNAPSHOT_PRINTED:
        return

    sample_id = sample.get("id", sample.get("sample_id", "N/A"))
    image_path = sample.get("image")
    question_raw = extract_question(sample)
    gt_raw = extract_ground_truth(sample)
    question_pre = _normalize_question_for_chat(question_raw)

    image_status = {
        "path": image_path,
        "exists": bool(image_path) and os.path.exists(image_path),
        "loaded": False,
    }
    pil_image = None
    if image_status["exists"]:
        try:
            with Image.open(image_path) as img:
                pil_image = img.convert("RGB")
            image_status["loaded"] = True
            image_status["pil_mode"] = pil_image.mode
            image_status["pil_size"] = tuple(pil_image.size)
        except Exception as exc:
            image_status["warning"] = f"image load failed: {type(exc).__name__}: {exc}"
    else:
        image_status["warning"] = "image path missing or not found"

    q_ids = _safe_encode(tokenizer, question_pre)
    q_decode = _safe_decode(tokenizer, q_ids)
    gt_ids = _safe_encode(tokenizer, gt_raw)
    gt_decode = _safe_decode(tokenizer, gt_ids)

    processor_summary = {"available": False}
    if model is not None:
        processor = _get_chat_processor(model, _resolve_max_slice_nums(model, None))
        if processor is not None:
            processor_summary["available"] = True
            processor_summary["type"] = type(processor).__name__
            image_processor = getattr(processor, "image_processor", None)
            processor_summary["has_image_processor"] = image_processor is not None
            if image_processor is not None and pil_image is not None:
                try:
                    processed = image_processor(images=pil_image, return_tensors="pt")
                    if isinstance(processed, dict):
                        processor_summary["image_processor_outputs"] = {
                            k: _summarize_tensor_like(v) for k, v in processed.items()
                        }
                    else:
                        processor_summary["image_processor_outputs"] = _summarize_tensor_like(processed)
                except Exception as exc:
                    processor_summary["image_processor_warning"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

    print("\n" + "=" * 90, flush=True)
    print(f"[{stage_tag}SampleDebug] First eval sample snapshot (printed once before first eval).", flush=True)
    print(f"[{stage_tag}SampleDebug] Stage 1: Raw sample", flush=True)
    print(f"  - id: {sample_id}", flush=True)
    print(f"  - image: {image_status}", flush=True)
    print(f"  - user_text_raw: {repr(_truncate_for_log(question_raw, 1800))}", flush=True)
    print(f"  - assistant_text_raw(gt): {repr(_truncate_for_log(gt_raw, 1800))}", flush=True)
    print(f"[{stage_tag}SampleDebug] Stage 2: Preprocessed eval text", flush=True)
    print(f"  - question_preprocessed: {repr(_truncate_for_log(question_pre, 1800))}", flush=True)
    print(f"[{stage_tag}SampleDebug] Stage 3: Tokenizer/processor view", flush=True)
    print(f"  - question_token_count: {len(q_ids)}", flush=True)
    print(f"  - question_decode: {repr(_truncate_for_log(q_decode, 1800))}", flush=True)
    print(f"  - gt_token_count: {len(gt_ids)}", flush=True)
    print(f"  - gt_decode: {repr(_truncate_for_log(gt_decode, 1800))}", flush=True)
    print(f"  - processor_summary: {processor_summary}", flush=True)
    print("=" * 90 + "\n", flush=True)

    _EVAL_SAMPLE_SNAPSHOT_PRINTED = True


def normalize_label(text: str) -> str:
    """Lowercase and strip whitespace/punctuation for fuzzy label matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_letter_to_label(label_set: List[str]) -> Dict[str, str]:
    letter_to_label: Dict[str, str] = {}
    for lbl in label_set:
        letter = _extract_label_letter_from_label(lbl)
        if letter and letter not in letter_to_label:
            letter_to_label[letter] = lbl
    return letter_to_label


def extract_prediction_letter(
    pred: str,
    label_set: Optional[List[str]] = None,
    question: Optional[str] = None,
) -> Optional[str]:
    """
    Recover the intended option letter from a free-form generation.

    Preference order:
    1. Leading option-letter forms such as `B`, `(B) ...`, `B) ...`
    2. Standalone uppercase letters elsewhere in the response
    3. Option-name text matched against the current question's option table
    """
    pred_text = "" if pred is None else str(pred).strip()
    if not pred_text:
        return None

    valid_letters = None
    if label_set:
        valid_letters = set(_build_letter_to_label(label_set))
    question_options = parse_question_options(question)
    if question_options:
        option_letters = set(question_options)
        valid_letters = option_letters if valid_letters is None else (valid_letters & option_letters)

    m = _ANSWER_LETTER_RE.search(pred_text)
    pred_letter = m.group(1).upper() if m else None
    if pred_letter and (not valid_letters or pred_letter in valid_letters):
        return pred_letter

    m = _LEADING_LETTER_RE.match(pred_text)
    pred_letter = m.group(1).upper() if m else None
    if pred_letter and (not valid_letters or pred_letter in valid_letters):
        return pred_letter

    pred_letter = _extract_label_letter_from_label(pred_text)
    if pred_letter and (not valid_letters or pred_letter in valid_letters):
        return pred_letter

    if valid_letters:
        for m in re.finditer(r"\b([A-Z])\b", pred_text.upper()):
            candidate = m.group(1)
            if candidate in valid_letters:
                return candidate

    if question_options:
        pred_norm = normalize_label(pred_text)
        for letter, option_name in question_options.items():
            if valid_letters and letter not in valid_letters:
                continue
            option_norm = normalize_label(option_name)
            if not option_norm:
                continue
            if pred_norm == option_norm or option_norm in pred_norm:
                return letter

    return None


def match_prediction_to_label(
    pred: str,
    label_set: List[str],
    question: Optional[str] = None,
) -> str:
    """
    Match a free-form model prediction to the closest canonical label.
    Tries exact match first, then option-letter mapping, then conservative
    substring matching.
    """
    pred_text = "" if pred is None else str(pred).strip()
    pred_norm = normalize_label(pred_text)
    unk_token = "UNK"
    if not pred_norm:
        return unk_token

    # 1. Exact match (case-insensitive)
    for lbl in label_set:
        if normalize_label(lbl) == pred_norm:
            return lbl

    # 2. Option-letter mapping (works for both label formats: "E" and "(E) xxx")
    letter_to_label = _build_letter_to_label(label_set)
    pred_letter = extract_prediction_letter(
        pred_text,
        label_set=label_set,
        question=question,
    )
    if pred_letter and pred_letter in letter_to_label:
        return letter_to_label[pred_letter]

    label_norms = [normalize_label(lbl) for lbl in label_set]
    single_letter_labels = all(
        (len(n) == 1 and n.isalpha()) for n in label_norms if n
    )

    # 3. Canonical label appears as substring of prediction.
    # Skip 1-char canonical labels to avoid accidental letter hits.
    for lbl, lbl_norm in zip(label_set, label_norms):
        if lbl_norm and len(lbl_norm) >= 2 and lbl_norm in pred_norm:
            return lbl

    if single_letter_labels and len(pred_norm) == 1 and pred_norm.isalpha():
        upper = pred_norm.upper()
        if upper in letter_to_label:
            return letter_to_label[upper]

    # 4. Prediction appears as substring of canonical label.
    # Disable for single-letter label sets and very short predictions.
    if len(pred_norm) >= 3:
        for lbl, lbl_norm in zip(label_set, label_norms):
            if lbl_norm and pred_norm in lbl_norm:
                return lbl

    # 5. Fallback: return as-is
    return unk_token


def classify_prediction_error(
    raw_output: Optional[str],
    parsed_pred: Optional[str],
    ground_truth: Optional[str],
) -> str:
    """
    Classify prediction outcome into one of:
    - correct
    - parse_error
    - empty_output
    - wrong_prediction
    """
    raw_text = "" if raw_output is None else str(raw_output).strip()
    pred_text = "" if parsed_pred is None else str(parsed_pred).strip()
    gt_text = "" if ground_truth is None else str(ground_truth).strip()

    if pred_text == gt_text and pred_text:
        return "correct"
    if not raw_text:
        return "empty_output"
    if not pred_text or normalize_label(pred_text) == normalize_label("UNK"):
        return "parse_error"
    return "wrong_prediction"


def summarize_prediction_errors(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        "correct": 0,
        "parse_error": 0,
        "empty_output": 0,
        "wrong_prediction": 0,
    }
    for rec in records:
        et = rec.get("error_type")
        if et in counts:
            counts[et] += 1
    total = len(records)
    summary = {
        "total": total,
        "correct": counts["correct"],
        "parse_error": counts["parse_error"],
        "empty_output": counts["empty_output"],
        "wrong_prediction": counts["wrong_prediction"],
        "ratios": {
            k: (counts[k] / total if total > 0 else 0.0)
            for k in counts
        },
    }
    return summary


def _extract_label_letter_from_label(label: str) -> Optional[str]:
    if label is None:
        return None
    text = str(label).strip().upper()
    if not text:
        return None
    m = re.match(r"^\(?([A-Z])\)", text)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z])(?:\s|$|[\.\):\-])", text)
    if m:
        return m.group(1)
    return None


def _extract_first_step_logits_from_response(obj: Any):
    if torch.is_tensor(obj):
        if obj.ndim == 2 and obj.size(0) >= 1:
            return obj[0].float().detach().cpu().numpy()
        if obj.ndim == 1:
            return obj.float().detach().cpu().numpy()
        return None

    scores = None
    if isinstance(obj, dict):
        scores = obj.get("scores", None)
        if scores is None and "logits" in obj:
            logits = obj.get("logits")
            if torch.is_tensor(logits):
                if logits.ndim >= 2:
                    return logits[0].float().detach().cpu().numpy()
                if logits.ndim == 1:
                    return logits.float().detach().cpu().numpy()
    elif hasattr(obj, "scores"):
        scores = getattr(obj, "scores", None)
    elif hasattr(obj, "logits"):
        logits = getattr(obj, "logits", None)
        if torch.is_tensor(logits):
            if logits.ndim >= 2:
                return logits[0].float().detach().cpu().numpy()
            if logits.ndim == 1:
                return logits.float().detach().cpu().numpy()
    if scores is None or not isinstance(scores, (list, tuple)) or len(scores) == 0:
        return None
    first = scores[0]
    if not torch.is_tensor(first):
        return None
    if first.ndim == 2 and first.size(0) >= 1:
        return first[0].float().detach().cpu().numpy()
    if first.ndim == 1:
        return first.float().detach().cpu().numpy()
    return None


def _find_first_step_logits(chat_response: Any):
    direct = _extract_first_step_logits_from_response(chat_response)
    if direct is not None:
        return direct

    if isinstance(chat_response, (list, tuple)):
        for item in chat_response:
            found = _find_first_step_logits(item)
            if found is not None:
                return found
        return None

    if isinstance(chat_response, dict):
        for value in chat_response.values():
            found = _find_first_step_logits(value)
            if found is not None:
                return found
    return None


def _forward_first_step_logits_qwen2_vl(
    model,
    question: str,
    image_path: Optional[str] = None,
):
    try:
        inputs, _ = _prepare_qwen2_vl_inputs(
            model=model,
            question=question,
            image_path=image_path,
            device="cuda",
        )
    except Exception:
        return None
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or not torch.is_tensor(logits) or logits.ndim < 3:
            return None
        return logits[0, -1].float().detach().cpu().numpy()
    except Exception:
        return None


def _chat_first_step_logits(
    model,
    tokenizer,
    question: str,
    image_path: Optional[str] = None,
    max_slice_nums: Optional[int] = None,
):
    if infer_model_family_from_model(model) == MODEL_FAMILY_QWEN2_VL:
        return _forward_first_step_logits_qwen2_vl(
            model=model,
            question=question,
            image_path=image_path,
        )
    normalized_question = _normalize_question_for_chat(question)
    resolved_max_slice_nums = _resolve_max_slice_nums(model, max_slice_nums)
    _sync_chat_runtime_config(model, resolved_max_slice_nums)
    processor = _get_chat_processor(model, resolved_max_slice_nums)
    kwargs_base = {
        "tokenizer": tokenizer,
        "max_new_tokens": 1,
        "sampling": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if resolved_max_slice_nums is not None:
        kwargs_base["max_slice_nums"] = resolved_max_slice_nums
    if processor is not None:
        kwargs_base["processor"] = processor
    has_image = image_path is not None and os.path.exists(image_path)
    candidates = []
    if has_image:
        with Image.open(image_path) as _img:
            image = _img.convert("RGB")
        candidates.extend([
            # Preferred inline multimodal format.
            dict(image=None, msgs=[{"role": "user", "content": [image, normalized_question]}], **kwargs_base),
            # Compatibility fallback: image as separate argument.
            dict(image=image, msgs=[{"role": "user", "content": normalized_question}], **kwargs_base),
            dict(image=image, msgs=[{"role": "user", "content": normalized_question}], tokenizer=tokenizer, max_new_tokens=1, sampling=False, return_logits=True),
            dict(image=image, msgs=[{"role": "user", "content": normalized_question}], tokenizer=tokenizer, max_new_tokens=1, output_scores=True),
            dict(image=image, msgs=[{"role": "user", "content": normalized_question}], tokenizer=tokenizer, max_new_tokens=1),
        ])
    else:
        candidates.extend([
            dict(image=None, msgs=[{"role": "user", "content": normalized_question}], **kwargs_base),
            dict(image=None, msgs=[{"role": "user", "content": normalized_question}], tokenizer=tokenizer, max_new_tokens=1, sampling=False, return_logits=True),
            dict(image=None, msgs=[{"role": "user", "content": normalized_question}], tokenizer=tokenizer, max_new_tokens=1, output_scores=True),
            dict(msgs=[{"role": "user", "content": normalized_question}], tokenizer=tokenizer, max_new_tokens=1, output_scores=True),
        ])

    for kwargs in candidates:
        if resolved_max_slice_nums is not None:
            kwargs.setdefault("max_slice_nums", resolved_max_slice_nums)
        if processor is not None:
            kwargs.setdefault("processor", processor)

    for kwargs in candidates:
        try:
            response = model.chat(**kwargs)
            logits = _find_first_step_logits(response)
            if logits is not None:
                return logits
        except Exception:
            continue
    return None


def _build_letter_token_ids(tokenizer, letters: List[str]) -> Dict[str, List[int]]:
    letter_to_ids: Dict[str, List[int]] = {}
    for letter in letters:
        cands = [letter, f" {letter}"]
        ids = set()
        for cand in cands:
            try:
                toks = tokenizer.encode(cand, add_special_tokens=False)
            except Exception:
                toks = []
            if len(toks) == 1:
                ids.add(int(toks[0]))
        if not ids:
            try:
                tid = tokenizer.convert_tokens_to_ids(letter)
                if isinstance(tid, int) and tid >= 0:
                    ids.add(int(tid))
            except Exception:
                pass
        letter_to_ids[letter] = sorted(ids)
    return letter_to_ids


def generate_label_probability_scores(
    model,
    tokenizer,
    question: str,
    label_set: List[str],
    image_path: Optional[str] = None,
    max_new_tokens: int = 1,
    max_slice_nums: Optional[int] = None,
    device: str = "cuda",
    fallback_text: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Return true per-class scores from model generation logits (first decoding step).
    If logits are unavailable from the current model.chat implementation, return None.
    """
    if not label_set:
        return []
    letters = []
    for lbl in label_set:
        letter = _extract_label_letter_from_label(lbl)
        letters.append(letter if letter is not None else "")
    valid_letters = sorted({x for x in letters if x})
    if not valid_letters:
        return None

    scoring_prompt = (
        f"{question}\n"
        f"Answer with ONLY one uppercase option letter from {valid_letters}. "
        f"Do not output anything else."
    )
    logits = _chat_first_step_logits(
        model=model,
        tokenizer=tokenizer,
        question=scoring_prompt,
        image_path=image_path,
        max_slice_nums=max_slice_nums,
    )
    if logits is None:
        return None

    token_map = _build_letter_token_ids(tokenizer, valid_letters)
    class_logits = []
    for letter in letters:
        if not letter:
            class_logits.append(-1e9)
            continue
        token_ids = token_map.get(letter, [])
        picked = []
        for tid in token_ids:
            if 0 <= tid < len(logits):
                picked.append(float(logits[tid]))
        class_logits.append(max(picked) if picked else -1e9)

    arr = np.array(class_logits, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return None
    arr[~finite_mask] = -1e9
    m = float(np.max(arr))
    exp_arr = np.exp(arr - m)
    s = float(np.sum(exp_arr))
    if s <= 0.0 or not np.isfinite(s):
        return None
    probs = (exp_arr / s).tolist()
    return [float(x) for x in probs]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    predictions: List[str],
    labels: List[str],
    label_names: Optional[List[str]] = None,
    score_matrix: Optional[List[List[float]]] = None,
) -> Dict:
    """
    Compute classification metrics.

    Args:
        predictions: Model prediction strings (already matched to labels).
        labels:      Ground-truth label strings.
        label_names: Optional canonical label list for ordering.
    Returns:
        Dict with accuracy/f1 and AUC (OvR) computed from continuous per-class scores.
    """
    if label_names is None:
        label_names = sorted(set(labels))

    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    per_class = classification_report(
        labels, predictions, labels=label_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(labels, predictions, labels=label_names).tolist()

    auc_ovr_macro = None
    auc_ovr_weighted = None
    auc_note = "auc_unavailable"
    auc_valid_rows = 0
    auc_total_rows = len(score_matrix) if score_matrix is not None else 0
    auc_present_classes = 0
    auc_present_labels: List[str] = []
    auc_skipped_rows = {
        "label_oov": 0,
        "row_type_or_len_mismatch": 0,
        "row_sum_nonpositive": 0,
    }
    try:
        if score_matrix is not None:
            label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}
            valid_true_idx = []
            valid_scores = []
            for y_true, row in zip(labels, score_matrix):
                if y_true not in label_to_idx:
                    auc_skipped_rows["label_oov"] += 1
                    continue
                if not isinstance(row, (list, tuple)) or len(row) != len(label_names):
                    auc_skipped_rows["row_type_or_len_mismatch"] += 1
                    continue
                row_vals = []
                for v in row:
                    try:
                        fv = float(v)
                    except Exception:
                        fv = 0.0
                    if not np.isfinite(fv):
                        fv = 0.0
                    row_vals.append(max(0.0, fv))
                s = float(sum(row_vals))
                if s <= 0.0:
                    auc_skipped_rows["row_sum_nonpositive"] += 1
                    continue
                row_vals = [x / s for x in row_vals]
                valid_true_idx.append(label_to_idx[y_true])
                valid_scores.append(row_vals)

            auc_valid_rows = len(valid_true_idx)
            if len(valid_true_idx) >= 2:
                y_true_idx = valid_true_idx
                y_score = np.array(valid_scores, dtype=np.float64)
                all_classes = list(range(len(label_names)))
                present_classes = sorted(set(y_true_idx))
                auc_present_classes = len(present_classes)
                auc_present_labels = [label_names[i] for i in present_classes]
                if len(present_classes) >= 2:
                    y_true_bin = label_binarize(y_true_idx, classes=all_classes)[:, present_classes]
                    y_score = y_score[:, present_classes]
                    auc_ovr_macro = roc_auc_score(y_true_bin, y_score, average="macro")
                    auc_ovr_weighted = roc_auc_score(y_true_bin, y_score, average="weighted")
                    auc_note = "ovr_from_model_logits_scores"
                else:
                    auc_note = "auc_unavailable_single_class"
            else:
                auc_note = "auc_unavailable_insufficient_samples"
        else:
            auc_note = "auc_requires_score_matrix"
    except Exception as e:
        auc_note = f"auc_unavailable_runtime_error:{type(e).__name__}"

    return {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "auc_ovr_macro": round(float(auc_ovr_macro), 4) if auc_ovr_macro is not None else None,
        "auc_ovr_weighted": round(float(auc_ovr_weighted), 4) if auc_ovr_weighted is not None else None,
        "auc_note": auc_note,
        "auc_valid_rows": auc_valid_rows,
        "auc_total_rows": auc_total_rows,
        "auc_present_classes": auc_present_classes,
        "auc_present_labels": auc_present_labels,
        "auc_skipped_rows": auc_skipped_rows,
        "per_class_report": per_class,
        "confusion_matrix": cm,
        "label_names": label_names,
        "num_samples": len(labels),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(
    cm: List[List[int]],
    label_names: List[str],
    output_path: str,
) -> None:
    """Save a confusion-matrix heatmap as PNG. Silently skips if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(5, len(label_names) - 1)))
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(label_names)),
            yticks=range(len(label_names)),
            xticklabels=label_names,
            yticklabels=label_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                )
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[eval] Confusion matrix saved → {output_path}")
    except ImportError:
        print("[eval] matplotlib not available; skipping confusion matrix plot.")
