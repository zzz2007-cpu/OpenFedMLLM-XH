import types
import torch
from torch import nn

from .autoregressive_hacker import AutoRegressiveHacker


class DifferenceHacker(AutoRegressiveHacker):

    def __init__(self, model, ref_model, tokenizer):
        super(DifferenceHacker, self).__init__(model, tokenizer)
        self.ref_model = ref_model
        self.kv_cache = {'cur_model': None, 'ref_model': None}

    def to(self, device):
        self.hacker_model.to(device)
        self.ref_model.to(device)
        self.device = device

    def nucleus_sampling(self, logits, top_p=0.6):
        """
            Sample from logits with Nucleus Sampling
        """
        softmax = nn.Softmax(dim=-1)
        probs = softmax(logits)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < top_p
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1, )), nucleus[..., :-1]], dim=-1)
        sorted_probs[~nucleus] = 0.

        final_probs = torch.zeros_like(sorted_probs)
        final_probs.scatter_(-1, indices, sorted_probs)
        return final_probs

    def generate(
        self,
        input_ids,
        max_new_tokens=512,
        temperature=0.95,
        top_p=0.6,
        do_sample=True,
        repetition_penalty=1.2,
        **kwargs
    ):

        def new_forward(_self, input_ids, **kwargs):
            attention_mask = input_ids.ne(0)

            # if self.kv_cache is not None:
            if self.kv_cache['cur_model'] is not None:
                input_ids = input_ids[:, -1:]

            output = orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.kv_cache['cur_model'],
                return_dict=True,
                use_cache=True,
            )

            ref_output = ref_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.kv_cache['cur_model'],
                return_dict=True,
                use_cache=True
            )

            method = "sfm+nucleus"
            if method == "raw":
                output.logits = output.logits - ref_output.logits
            elif method == "sfm":
                # softmax, new logits
                sfm_temperature = 1
                tmp = output.logits * torch.nn.functional.softmax(
                    (output.logits - ref_output.logits) / sfm_temperature, dim=-1
                )
                tmp = torch.norm(output.logits) / torch.norm(tmp) * tmp
                output.logits = tmp
            elif method == "sfm+nucleus":
                # softmax, new logits
                sfm_temperature = 1
                new_output_logits = self.nucleus_sampling(output.logits, top_p=0.8)
                tmp = new_output_logits * torch.nn.functional.softmax(
                    (output.logits - ref_output.logits) / sfm_temperature, dim=-1
                )
                tmp = torch.norm(new_output_logits) / torch.norm(tmp) * tmp
                output.logits = tmp

            self.kv_cache['cur_model'] = output.past_key_values
            self.kv_cache['ref_model'] = ref_output.past_key_values
            return output

        # Reset the kv cache.
        self.kv_cache = {'cur_model': None, 'ref_model': None}
        orig_forward = self.hacker_model.forward
        self.hacker_model.forward = types.MethodType(new_forward, self.hacker_model)
        ref_forward = self.ref_model.forward

        # Generate.
        ret = self.hacker_model.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            **kwargs
        )
        self.hacker_model.forward = orig_forward
        return ret
