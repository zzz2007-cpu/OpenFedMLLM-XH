import os
import importlib.util
import math
from types import MethodType
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover - compatibility for older transformers
    AutoModelForVision2Seq = None
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ..adapters import (
    MODEL_FAMILY_QWEN2_VL,
    resolve_model_family,
    select_effective_lora_target_modules,
)


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return {"Total": all_param, "Trainable": trainable_params}


def validate_quantization_dependencies(model_args, lora_args):
    requires_bnb = lora_args.q_lora or "int4" in str(model_args.model_name_or_path).lower() or "int8" in str(model_args.model_name_or_path).lower()
    if not requires_bnb:
        return
    missing = []
    if importlib.util.find_spec("accelerate") is None:
        missing.append("accelerate")
    if importlib.util.find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes")
    if missing:
        raise ImportError(
            "Quantized model loading requires missing dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install accelerate bitsandbytes`, "
            + "or disable q_lora / switch to a non-int4 model."
        )


def patch_minicpmv_get_vllm_embedding(model):
    """
    MiniCPM-V compatibility patch:
    avoid in-place writes on leaf tensors requiring grad in training mode.
    """
    if not hasattr(model, "get_vllm_embedding"):
        return
    if getattr(model, "_openfed_safe_vllm_patch", False):
        return
    if not (hasattr(model, "llm") and hasattr(model, "vpm") and hasattr(model, "resampler")):
        return

    def safe_get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                batch_size, seq_len, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(batch_size, 3, -1, seq_len)

                patch_attn_mask = torch.zeros((batch_size, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(batch_size):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if batch_size > vision_batch_size:
                    hs = []
                    for i in range(0, batch_size, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        # Keep original shape/dtype/device semantics, but avoid in-place edits.
        bs = len(data["input_ids"])
        rows = []
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            cur_vllm_emb = vllm_embedding[i]
            if len(cur_vs_hs) > 0:
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)
                    scatter_index = image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1])
                    scatter_src = cur_vs_hs.view(-1, cur_vs_hs.shape[-1])
                    # Safe replacement of `scatter_`: clone + out-of-place scatter.
                    cur_vllm_emb = cur_vllm_emb.clone().scatter(0, scatter_index, scatter_src)
                elif self.training:
                    # Safe replacement of `+=` on a view.
                    cur_vllm_emb = cur_vllm_emb + cur_vs_hs[0].mean() * 0
            rows.append(cur_vllm_emb)

        vllm_embedding = torch.stack(rows, dim=0)
        return vllm_embedding, vision_hidden_states

    model.get_vllm_embedding = MethodType(safe_get_vllm_embedding, model)
    model._openfed_safe_vllm_patch = True
    print("[MiniCPMVPatch] Applied safe get_vllm_embedding patch (no in-place scatter_/+= on grad leaf).")


def _freeze_if_exists(model, attr_name):
    module = getattr(model, attr_name, None)
    if module is None:
        return False
    try:
        module.requires_grad_(False)
        return True
    except Exception:
        return False


def _bind_processor_to_model(model, processor):
    if processor is None:
        return
    try:
        model.processor = processor
    except Exception:
        pass


def build_model_and_tokenizer(model_args, training_args, lora_args):
    validate_quantization_dependencies(model_args, lora_args)
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if (lora_args.q_lora and ddp) else None
    model_family = resolve_model_family(
        model_args.model_name_or_path,
        llm_type=getattr(training_args, "llm_type", None),
        explicit_family=getattr(model_args, "model_family", None),
    )
    trust_remote_code = bool(getattr(model_args, "trust_remote_code", True))
    attn_implementation = getattr(model_args, "attn_implementation", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=trust_remote_code,
        cache_dir=training_args.cache_dir,
    )

    model_kwargs = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=compute_dtype,
        device_map=device_map,
        cache_dir=training_args.cache_dir,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if model_family == MODEL_FAMILY_QWEN2_VL:
        if AutoModelForVision2Seq is None:
            raise ImportError(
                "Qwen2-VL requires a newer transformers version with "
                "AutoModelForVision2Seq support."
            )
        processor_kwargs = {}
        processor_min_pixels = getattr(model_args, "processor_min_pixels", None)
        processor_max_pixels = getattr(model_args, "processor_max_pixels", None)
        if processor_min_pixels is not None:
            processor_kwargs["min_pixels"] = int(processor_min_pixels)
        if processor_max_pixels is not None:
            processor_kwargs["max_pixels"] = int(processor_max_pixels)
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=training_args.cache_dir,
            **processor_kwargs,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        _bind_processor_to_model(model, processor)
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        patch_minicpmv_get_vllm_embedding(model)
    model._openfed_model_family = model_family

    if getattr(training_args, "enable_audio", False) and not hasattr(model, "apm"):
        raise ValueError("Audio training is enabled, but the current model does not provide an audio module.")

    if not training_args.tune_vision:
        vision_frozen = False
        vision_frozen |= _freeze_if_exists(model, "vpm")
        vision_frozen |= _freeze_if_exists(model, "visual")
        vision_frozen |= _freeze_if_exists(model, "vision_tower")
        if not vision_frozen:
            for name, param in model.named_parameters():
                lowered = name.lower()
                if "visual" in lowered or "vision" in lowered:
                    param.requires_grad = False
    if not training_args.tune_llm:
        llm_frozen = _freeze_if_exists(model, "llm")
        if not llm_frozen:
            for name, param in model.named_parameters():
                lowered = name.lower()
                if "visual" in lowered or "vision" in lowered:
                    continue
                param.requires_grad = False

    if training_args.use_lora:
        if training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
        for _, param in model.named_parameters():
            param.requires_grad = False
        effective_target_modules = select_effective_lora_target_modules(
            model=model,
            requested_target_modules=lora_args.lora_target_modules,
            model_family=model_family,
        )
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=effective_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            # Strict pure-LoRA mode: do not keep any full base module trainable.
            modules_to_save=None,
        )
        if not hasattr(model, "get_input_embeddings") and hasattr(model, "llm"):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        model = get_peft_model(model, lora_config)
        # Safety fuse: guarantee only LoRA adapter tensors remain trainable.
        for name, param in model.named_parameters():
            if "lora_" not in name.lower():
                param.requires_grad = False
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    return model, tokenizer
