from peft import LoraConfig, get_peft_model


def add_wrapper(model, name=None, **kwargs):
    if name is None:
        return model
    if name.lower() == "lora":
        # Strict pure-LoRA mode for mllm wrapper path.
        kwargs["modules_to_save"] = None
        config = LoraConfig(**kwargs)
        model = get_peft_model(model, config)
        for param_name, param in model.named_parameters():
            if "lora_" not in param_name.lower():
                param.requires_grad = False
        return model
    return model
