import os
import torch
from .sft_mllm_fedavg_trainer import (
    CPMTrainer,
    amp,
    is_sagemaker_mp_enabled,
    smp_forward_backward,
)


class CPMTrainerScaffold(CPMTrainer):
    """
    SCAFFOLD trainer for MLLM SFT.

    It injects the control-variate correction term into local optimization:
      grad <- grad + c - c_i
    and exposes utility for updating client control variates after local training.
    """

    def __init__(
        self,
        global_auxiliary,
        local_auxiliary,
        scaffold_lr,
        scaffold_eps=1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_auxiliary = global_auxiliary
        self.local_auxiliary = local_auxiliary
        self.scaffold_lr = scaffold_lr
        self.scaffold_eps = scaffold_eps
        self._last_scaffold_corr_norm = None
        self._last_scaffold_match_count = 0
        self._debug_scaffold = str(os.environ.get("OPENFED_DEBUG_SCAFFOLD", "0")).lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("modules_to_save.", "").replace("module.", "").replace("default.", "")

    def _iter_scaffold_params(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            key = self._normalize_name(name)
            if key not in self.global_auxiliary or key not in self.local_auxiliary:
                continue
            yield key, param

    def _apply_scaffold_grad_correction(self, model):
        # With gradient accumulation, each micro-step contributes 1/accum of
        # the final step-level correction so the accumulated correction equals
        # (c - c_i) once per optimizer update.
        grad_acc = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        corr_scale = 1.0 / float(grad_acc)

        corr_sq = 0.0
        matched = 0
        for key, param in self._iter_scaffold_params(model):
            if param.grad is None:
                continue
            server_c = self.global_auxiliary[key].to(device=param.grad.device, dtype=param.grad.dtype)
            client_c = self.local_auxiliary[key].to(device=param.grad.device, dtype=param.grad.dtype)
            corr = server_c - client_c
            param.grad.add_(corr, alpha=corr_scale)
            corr_sq += float(torch.sum(corr.float() * corr.float()).item())
            matched += 1

        self._last_scaffold_corr_norm = corr_sq ** 0.5 if matched > 0 else 0.0
        self._last_scaffold_match_count = matched
        if self._debug_scaffold and matched == 0:
            print("[SCAFFOLD][WARN] No trainable params matched control variates.", flush=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Keep reported/scalar loss as task loss only. The SCAFFOLD correction
        # is injected directly into gradients in training_step.
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled() and smp_forward_backward is not None:
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        del inputs
        torch.cuda.empty_cache()
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.use_apex and amp is not None:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        self._apply_scaffold_grad_correction(model)

        grad_norm = None
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_value = param.grad.detach().float().norm().item()
            grad_norm = grad_value if grad_norm is None else grad_norm + grad_value ** 2
        self._last_grad_norm = (grad_norm ** 0.5) if grad_norm is not None else None
        return loss.detach() / self.args.gradient_accumulation_steps

    def get_federated_metrics(self):
        metrics = super().get_federated_metrics()
        metrics.update(
            {
                "scaffold_corr_norm": self._last_scaffold_corr_norm,
                "scaffold_matched_params": self._last_scaffold_match_count,
            }
        )
        return metrics

    def update_auxiliary(self, global_state_before, local_state_after, local_steps):
        """
        Update local control variate c_i and return:
          - new local control variate dict
          - delta control variate dict (c_i_new - c_i_old)
        """
        lr = max(float(self.scaffold_lr), float(self.scaffold_eps))
        steps = max(1, int(local_steps) if local_steps is not None else 1)

        new_local_auxiliary = {}
        auxiliary_delta = {}

        for key, local_c in self.local_auxiliary.items():
            local_c_fp32 = local_c.float()
            if key not in self.global_auxiliary:
                new_local_auxiliary[key] = local_c_fp32.clone()
                auxiliary_delta[key] = torch.zeros_like(local_c_fp32)
                continue
            if key not in global_state_before or key not in local_state_after:
                new_local_auxiliary[key] = local_c_fp32.clone()
                auxiliary_delta[key] = torch.zeros_like(local_c_fp32)
                continue

            # c_i^{t+1} = c_i^t - c^t + (w^t - w_i^{t+1}) / (K * lr)
            correction = (global_state_before[key].float() - local_state_after[key].float()) / (steps * lr)
            new_c = local_c_fp32 - self.global_auxiliary[key].float() + correction
            auxiliary_delta[key] = new_c - local_c_fp32
            new_local_auxiliary[key] = new_c

        if self._debug_scaffold:
            delta_sq = 0.0
            for key, delta in auxiliary_delta.items():
                delta_sq += float(torch.sum(delta.float() * delta.float()).item())
            print(
                f"[SCAFFOLD][ClientAux] steps={steps} lr={lr:.6g} "
                f"delta_c_l2={delta_sq ** 0.5:.6f}",
                flush=True,
            )

        self.local_auxiliary = new_local_auxiliary
        return new_local_auxiliary, auxiliary_delta
