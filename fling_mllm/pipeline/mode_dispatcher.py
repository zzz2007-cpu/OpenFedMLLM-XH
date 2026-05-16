import copy
import os

from ..config.arguments import (
    DataArguments,
    FedArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments,
)
from .baseline_runner import (
    FinalModelEvalHook,
    run_centralized_baseline,
    run_local_only_baseline,
)
from .generic_model_mllm_pipeline import run_federated_finetune
from .tiny_hateful_cpu_pipeline import run_tiny_hateful_memes_fedavg_cpu


SUPPORTED_MODES = {"federated", "local_only", "centralized"}


def _to_plain_dict(section):
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    try:
        return dict(section)
    except Exception:
        return {}


def _get_section(exp_args, name: str):
    if isinstance(exp_args, dict):
        return exp_args.get(name, None)
    return getattr(exp_args, name, None)


def _normalize_mode(mode: str) -> str:
    if mode is None:
        return "federated"
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported mode={mode!r}. Supported modes: {sorted(SUPPORTED_MODES)}"
        )
    return normalized


def run_mode_from_config(exp_args, mode_override=None):
    model_args = ModelArguments(**_to_plain_dict(_get_section(exp_args, "model_args")))
    data_args = DataArguments(**_to_plain_dict(_get_section(exp_args, "data_args")))
    training_args = TrainingArguments(**_to_plain_dict(_get_section(exp_args, "training_args")))
    lora_args = LoraArguments(**_to_plain_dict(_get_section(exp_args, "lora_args")))
    fed_args = FedArguments(**_to_plain_dict(_get_section(exp_args, "fed_args")))
    eval_args = _to_plain_dict(_get_section(exp_args, "eval_args"))
    hooks = getattr(exp_args, "hooks", None)

    run_args = _to_plain_dict(_get_section(exp_args, "run_args"))
    mode = _normalize_mode(mode_override if mode_override is not None else run_args.get("mode", "federated"))
    if bool(run_args.get("append_mode_subdir", False)):
        training_args.output_dir = os.path.join(training_args.output_dir, mode)

    # Work on a detached copy so caller's in-memory config stays unchanged.
    training_args = copy.deepcopy(training_args)
    if mode == "federated":
        if str(model_args.model_name_or_path) == "tiny_hateful_memes_cpu":
            return run_tiny_hateful_memes_fedavg_cpu(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                fed_args=fed_args,
                eval_args=eval_args,
            )
        hook_list = list(hooks) if hooks is not None else []
        enable_final_eval = bool(run_args.get("enable_final_eval", True))
        if enable_final_eval and eval_args.get("eval_data_path"):
            hook_list.append(FinalModelEvalHook(eval_args=eval_args, output_dir=training_args.output_dir))
        return run_federated_finetune(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            lora_args=lora_args,
            fed_args=fed_args,
            hooks=hook_list,
        )
    if mode == "local_only":
        return run_local_only_baseline(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            lora_args=lora_args,
            fed_args=fed_args,
            eval_args=eval_args,
            hooks=hooks,
        )
    return run_centralized_baseline(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        lora_args=lora_args,
        fed_args=fed_args,
        eval_args=eval_args,
        hooks=hooks,
    )
