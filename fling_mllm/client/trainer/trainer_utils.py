from .sft_mllm_fedavg_trainer import CPMTrainer
from .sft_mllm_fedprox_trainer import CPMTrainerReg
from .sft_mllm_scaffold_trainer import CPMTrainerScaffold
from .sft_mllm_fednova_trainer import CPMTrainerFedNova


def get_mllm_trainer(name: str):
    mapper = {
        "mllm_sft_fedavg_trainer": CPMTrainer,
        "mllm_sft_fedprox_trainer": CPMTrainerReg,
        "mllm_sft_scaffold_trainer": CPMTrainerScaffold,
        "mllm_sft_fednova_trainer": CPMTrainerFedNova,
    }
    if name not in mapper:
        raise KeyError(f"Unknown trainer name: {name}")
    return mapper[name]
