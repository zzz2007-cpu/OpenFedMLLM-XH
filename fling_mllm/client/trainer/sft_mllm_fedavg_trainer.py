import os
import re
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.trainer import PreTrainedModel, TRAINING_ARGS_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME, unwrap_model, is_peft_available
try:
    from transformers.trainer import is_sagemaker_mp_enabled
except ImportError:
    def is_sagemaker_mp_enabled():
        return False
try:
    from transformers.trainer import smp_forward_only, smp_nested_concat, smp_forward_backward
except ImportError:
    smp_forward_only = None
    smp_nested_concat = None
    smp_forward_backward = None
try:
    from transformers.trainer import amp
except ImportError:
    amp = None
from transformers.trainer_pt_utils import nested_detach
from peft import PeftModel
import safetensors.torch
from ...adapters import is_minicpm_family


class CPMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_loss = None
        self._last_grad_norm = None
        self._last_logits_mean = None
        self._last_logits_std = None
        self._fed_initial_optimizer_lr = None

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # Force constant LR inside each client local training.
        if self.lr_scheduler is None:
            opt = optimizer if optimizer is not None else self.optimizer
            self.lr_scheduler = get_scheduler(
                name="constant",
                optimizer=opt,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def create_optimizer(self):
        super().create_optimizer()
        if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            self._fed_initial_optimizer_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        return self.optimizer

    def train(self, *args, **kwargs):
        round_idx = getattr(self, "fed_round_idx", -1)
        client_idx = getattr(self, "fed_client_idx", -1)
        configured_lr = float(getattr(self.args, "learning_rate", 0.0))
        # Ensure optimizer exists so we can log real optimizer LR before local steps.
        if self.optimizer is None:
            self.create_optimizer()
        optimizer_init_lr = None
        if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            optimizer_init_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        print(
            f"[ClientLR][start] round={round_idx} client={client_idx} "
            f"args.lr={configured_lr:.10g} optimizer.init.lr={optimizer_init_lr}",
            flush=True,
        )
        result = super().train(*args, **kwargs)
        optimizer_end_lr = None
        if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            optimizer_end_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        print(
            f"[ClientLR][end] round={round_idx} client={client_idx} "
            f"args.lr={configured_lr:.10g} optimizer.end.lr={optimizer_end_lr}",
            flush=True,
        )
        return result

    def get_federated_metrics(self):
        return {
            "loss": self._last_loss,
            "grad_norm": self._last_grad_norm,
            "logits_mean": self._last_logits_mean,
            "logits_std": self._last_logits_std,
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") if "labels" in inputs else None

        if is_minicpm_family(self.model):
            if not self.args.use_lora:
                outputs = self.model(data=inputs, use_cache=False)
            else:
                with self.model._enable_peft_forward_hooks(**inputs):
                    outputs = self.model.base_model(data=inputs, use_cache=False)
        else:
            outputs = model(**inputs, use_cache=False)
        if labels is not None:
            logits = outputs.logits
            if logits.ndim < 3:
                raise ValueError(f"Expected logits with shape [B, T, V], but got {tuple(logits.shape)}")
            if labels.ndim < 2:
                raise ValueError(f"Expected labels with shape [B, T], but got {tuple(labels.shape)}")
            if logits.size(1) < 2 or labels.size(1) < 2:
                loss = logits.sum() * 0.0
                self._last_logits_mean = logits.float().mean().item()
                self._last_logits_std = logits.float().std().item()
                self._last_loss = loss.detach().item()
                return (loss, outputs) if return_outputs else loss

            # Standard causal LM objective: predict token t+1 from position t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].long().contiguous().to(shift_logits.device)

            self._last_logits_mean = shift_logits.float().mean().item()
            self._last_logits_std = shift_logits.float().std().item()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            self._last_loss = loss.detach().item()
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            self._last_loss = loss.detach().item()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", []) if hasattr(self.model, "config") else []
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        with torch.no_grad():
            if is_sagemaker_mp_enabled() and smp_forward_only is not None and smp_nested_concat is not None:
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]
                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys) if isinstance(raw_outputs, dict) else raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"]) if isinstance(outputs, dict) else outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys) if isinstance(outputs, dict) else outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]
        if prediction_loss_only:
            return (loss, None, None)
        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
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
        grad_norm = None
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_value = param.grad.detach().float().norm().item()
            grad_norm = grad_value if grad_norm is None else grad_norm + grad_value ** 2
        self._last_grad_norm = (grad_norm ** 0.5) if grad_norm is not None else None
        return loss.detach() / self.args.gradient_accumulation_steps

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()
            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            else:
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


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
                loss += self.mu / 2 * self.mu_w * torch.norm(param - self.global_state[name]) ** 2
        return (loss, outputs) if return_outputs else loss
