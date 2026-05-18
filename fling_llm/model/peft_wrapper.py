from torch import nn
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


def wrap_lora(model: nn.Module, **lora_args) -> nn.Module:
    """
    Overview:
        Wrap the model with lora.
    Arguments:
        - model: The model to be wrapped.
    Returns:
        - model: The wrapped model with lora.
    """
    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, **lora_args)
    print(lora_args)
    q_lora = lora_args.pop("q_lora")
    # Strict pure-LoRA mode: never keep full base modules trainable.
    lora_args["modules_to_save"] = None
    peft_config = LoraConfig(**lora_args)
    if q_lora:
        model = prepare_model_for_kbit_training(model)

    print('LoRA Config:', peft_config)
    model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
        if "lora_" not in name.lower():
            param.requires_grad = False
    return model


name2wrapper = {'lora': wrap_lora}


def add_wrapper(model: nn.Module, **kwargs) -> nn.Module:
    method_name = kwargs.pop('name')
    if method_name is None:
        return model
    wrap_func = name2wrapper[method_name]
    return wrap_func(model, **kwargs)
