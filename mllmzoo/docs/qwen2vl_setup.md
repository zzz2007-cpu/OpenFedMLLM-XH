# Qwen2-VL-2B Dependency Notes

This project keeps MiniCPM-V and Qwen2-VL in one codebase. To reduce upgrade risk, use a narrow dependency band instead of unbounded upgrades.

## Recommended (for Qwen2-VL + MiniCPM-V coexistence)

- `transformers>=4.50.0,<5`
- `peft>=0.10.0`
- `accelerate>=0.30.0`
- `torch>=2.1`
- `torchvision` compatible with your installed torch

Optional:

- `flash-attn` only when CUDA stack matches your environment
- `bitsandbytes` only for `q_lora=True` or int4/int8 model checkpoints

## Conservative Fallback

If your cluster has strict CUDA constraints, keep existing torch/CUDA and only upgrade `transformers` + `peft` to the minimum versions above.

## Notes

- `qwen-vl-utils` is not required by this repository’s Qwen2-VL path.
- Keep `attn_implementation=None` by default unless you have validated flash-attn runtime compatibility.
- On CPU-only hosts, set `bf16=False` and `fp16=False`.
