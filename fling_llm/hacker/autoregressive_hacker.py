import types


class AutoRegressiveHacker:

    def __init__(self, model, tokenizer):
        self.hacker_model = model
        self.hacker_tokenizer = tokenizer
        self.device = self.hacker_model.device
        self.kv_cache = None

    def to(self, device):
        self.hacker_model.to(device)
        self.device = device

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
            attention_mask = input_ids.ne(0).to(self.device)
            if self.kv_cache is not None:
                input_ids = input_ids[:, -1:]

            output = orig_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.kv_cache,
                return_dict=True,
                use_cache=True,
            )
            self.kv_cache = output.past_key_values
            return output

        # Reset the kv cache.
        self.kv_cache = None
        orig_forward = self.hacker_model.forward
        self.hacker_model.forward = types.MethodType(new_forward, self.hacker_model)

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

    def hacking_generate(self, batch, **kwargs):
        return self.generate(batch, **kwargs)
