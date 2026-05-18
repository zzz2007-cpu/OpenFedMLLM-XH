import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='alpaca',
        # Default alpaca: 'tatsu-lab/alpaca'
        # Alpaca-GPT4: 'vicgalle/alpaca-gpt4'
        data_path='tatsu-lab/alpaca',
        tokenizer='Qwen/Qwen2.5-0.5B',
        max_len=4096,
        sample_method=dict(name='uniform', train_num=0, test_num=0)
    ),
    learn=dict(
        device='cuda:0',
        local_iters=500,
        global_eps=40,
        batch_size=1,
        trainer=dict(name='sft_fedavg_trainer'),
        hf_args=dict(
            bf16=True,
            tf32=True,
            gradient_accumulation_steps=8,
        )
    ),
    model=dict(
        model_path='Qwen/Qwen2.5-0.5B',
        pretrained=True,
        trust_remote_code=True,
        revision='main',
        torch_dtype=torch.bfloat16,
        # Check whether your machine support this.
        # attn_implementation="flash_attention_2"
        peft_config=dict(name='lora', r=8, lora_alpha=8, lora_dropout=0.1,
                         target_modules=['q_proj', 'v_proj']),
    ),
    client=dict(name='base_llm_client', client_num=5),
    server=dict(name='base_llm_server'),
    group=dict(name='base_group', aggregation_method='avg', include_non_param=False),
    other=dict(test_freq=1, logging_path='./logging/alpaca_fedavg_qwen2.5_0.5B')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)
