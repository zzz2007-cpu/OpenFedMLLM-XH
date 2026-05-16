import re
import torch
from .sft_mllm_fedavg_trainer import CPMTrainer


class CPMTrainerReg(CPMTrainer):
    def __init__(self, global_state, prox_mu, s_layer, mu_w, **kwargs):
        super().__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
        self.s_layer = s_layer
        self.mu_w = mu_w

    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values
        for name, param in model.named_parameters():
            name = name.replace("modules_to_save.", "").replace("module.", "").replace("default.", "")
            if not param.requires_grad:
                continue
            layer_num = re.search(r"\d+", name)
            reg_flag = False
            if layer_num:
                reg_flag = (int(layer_num.group()) >= self.s_layer) and (int(layer_num.group()) <= (27 - self.s_layer))
            if reg_flag and name in self.global_state:
                global_param = self.global_state[name].to(device=param.device, dtype=param.dtype)
                loss += self.mu / 2 * self.mu_w * torch.norm(param - global_param) ** 2
        return (loss, outputs) if return_outputs else loss
