from .sft_mllm_fedavg_trainer import CPMTrainer


class CPMTrainerFedNova(CPMTrainer):
    """
    FedNova trainer for MLLM SFT.

    It reuses the same local training loop as FedAvg and only exposes
    per-client normalization statistics required by server aggregation.
    """

    def __init__(self, fednova_eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.fednova_eps = fednova_eps

    def get_fednova_stats(self, train_result):
        local_steps = getattr(train_result, "global_step", None)
        if local_steps is None:
            local_steps = getattr(self.state, "global_step", 0)
        tau_eff = max(float(local_steps), float(self.fednova_eps))
        normalizer = max(tau_eff, float(self.fednova_eps))
        return {
            "local_steps": int(local_steps),
            "tau_eff": tau_eff,
            "normalizer": normalizer,
        }
