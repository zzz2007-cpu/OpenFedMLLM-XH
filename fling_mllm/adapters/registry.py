from typing import Optional


MODEL_FAMILY_MINICPM = "minicpm"
MODEL_FAMILY_QWEN2_VL = "qwen2_vl"


DEFAULT_MINICPM_LORA_TARGET_REGEX = r"llm\..*layers\.\d+\.self_attn\.(q_proj|v_proj)"


def _normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).strip().lower().replace("-", "_")


def resolve_model_family(
    model_name_or_path: Optional[str],
    llm_type: Optional[str] = None,
    explicit_family: Optional[str] = None,
) -> str:
    explicit = _normalize(explicit_family)
    if explicit in {MODEL_FAMILY_MINICPM, MODEL_FAMILY_QWEN2_VL}:
        return explicit

    llm = _normalize(llm_type)
    if llm in {MODEL_FAMILY_QWEN2_VL, "qwen2vl", "qwen2_vl_2b"}:
        return MODEL_FAMILY_QWEN2_VL
    if llm in {MODEL_FAMILY_MINICPM, "minicpm_v", "minicpmv"}:
        return MODEL_FAMILY_MINICPM

    name = _normalize(model_name_or_path)
    if "qwen2_vl" in name or "qwen2-vl" in str(model_name_or_path or "").lower():
        return MODEL_FAMILY_QWEN2_VL
    return MODEL_FAMILY_MINICPM


def infer_model_family_from_model(model) -> str:
    family = getattr(model, "_openfed_model_family", None)
    if family:
        return resolve_model_family(None, explicit_family=family)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        model_type = _normalize(getattr(cfg, "model_type", None))
        if model_type in {MODEL_FAMILY_QWEN2_VL, "qwen2vl"}:
            return MODEL_FAMILY_QWEN2_VL
        name_or_path = getattr(cfg, "_name_or_path", None)
        if name_or_path:
            return resolve_model_family(name_or_path)

    cls_name = type(model).__name__.lower()
    if "qwen2vl" in cls_name or "qwen2_vl" in cls_name:
        return MODEL_FAMILY_QWEN2_VL
    return MODEL_FAMILY_MINICPM


def is_minicpm_family(family_or_model) -> bool:
    if isinstance(family_or_model, str):
        return resolve_model_family(None, explicit_family=family_or_model) == MODEL_FAMILY_MINICPM
    return infer_model_family_from_model(family_or_model) == MODEL_FAMILY_MINICPM


def is_qwen2_vl_family(family_or_model) -> bool:
    if isinstance(family_or_model, str):
        return resolve_model_family(None, explicit_family=family_or_model) == MODEL_FAMILY_QWEN2_VL
    return infer_model_family_from_model(family_or_model) == MODEL_FAMILY_QWEN2_VL


def _infer_default_qwen2_vl_targets(model):
    candidate_suffixes = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    discovered = set()
    for module_name, _ in model.named_modules():
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in candidate_suffixes:
            discovered.add(leaf)

    preferred = [x for x in ["q_proj", "v_proj"] if x in discovered]
    if preferred:
        return preferred
    if discovered:
        return sorted(discovered)
    return ["q_proj", "v_proj"]


def select_effective_lora_target_modules(model, requested_target_modules, model_family: str):
    """
    Resolve LoRA target modules in a model-aware way.

    - Keep user-defined custom target modules untouched.
    - Auto-switch the default MiniCPM regex to Qwen2-VL-friendly targets.
    """
    family = resolve_model_family(None, explicit_family=model_family)
    requested = requested_target_modules

    if isinstance(requested, str):
        requested = requested.strip()
        if not requested:
            requested = None

    if requested is None or (isinstance(requested, str) and requested.lower() == "auto"):
        if family == MODEL_FAMILY_QWEN2_VL:
            return _infer_default_qwen2_vl_targets(model)
        return DEFAULT_MINICPM_LORA_TARGET_REGEX

    if family == MODEL_FAMILY_QWEN2_VL and requested == DEFAULT_MINICPM_LORA_TARGET_REGEX:
        return _infer_default_qwen2_vl_targets(model)
    return requested
