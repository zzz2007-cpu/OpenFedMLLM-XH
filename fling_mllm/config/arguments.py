from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2_6-int4")
    model_family: Optional[str] = field(default=None)
    trust_remote_code: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default=None)
    processor_min_pixels: Optional[int] = field(default=None)
    processor_max_pixels: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    eval_data_path: str = field(default=None)
    task_type: str = field(default="classification")
    data_format: str = field(default="auto")
    train_split: str = field(default="train")
    eval_split: str = field(default="eval")
    hateful_memes_root: Optional[str] = field(default=None)
    stat_setting: Optional[str] = field(default=None)
    modal_setting: Optional[str] = field(default=None)
    vqa_image_root: Optional[str] = field(default=None)
    vqa_prompt_template: Optional[str] = field(default=None)
    strict_image_path: Optional[bool] = field(default=True)
    max_train_samples_per_client: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default="../model/")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    vision_batch_size: Optional[int] = field(default=None)
    enable_audio: Optional[bool] = field(default=False)
    output_dir: str = field(default="../output/output_lora")


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedadagrad-fedprox-adaptive")
    mu_w: Optional[float] = field(default=0.1)
    s_layer: Optional[int] = field(default=4)
    num_rounds: Optional[int] = field(default=50)
    num_clients: Optional[int] = field(default=2)
    sample_clients: Optional[int] = field(default=2)
    split_strategy: Optional[str] = field(default="noniid")
    init_learning_rate: Optional[float] = field(default=0.01)
    outer_lr_schedule: Optional[str] = field(default="cosine")
    outer_lr_eta_min: Optional[float] = field(default=0.0)
    outer_lr_warmup_rounds: Optional[int] = field(default=None)
    prox_mu: Optional[float] = field(default=0.01)
    modality_num: Optional[float] = field(default=2)
    fedopt_tau: Optional[float] = field(default=1e-3)
    fedopt_eta: Optional[float] = field(default=1e-3)
    fedopt_beta1: Optional[float] = field(default=0.9)
    fedopt_beta2: Optional[float] = field(default=0.99)
    scaffold_server_lr: Optional[float] = field(default=1.0)
    scaffold_eps: Optional[float] = field(default=1e-12)
    fednova_server_lr: Optional[float] = field(default=1.0)
    fednova_eps: Optional[float] = field(default=1e-12)
    save_model_freq: Optional[int] = field(default=5)
