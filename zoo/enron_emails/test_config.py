import torch
from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='enron-email',
        data_path='./data/enron-emails',
        tokenizer='./data/Llama-2-7b-hf/',
        # tokenizer='meta-llama/Llama-2-7b-hf',
        max_len=2048,
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
        model_path='./data/Llama-2-7b-hf/',
        pretrained=True,
        trust_remote_code=True,
        revision='main',
        torch_dtype=torch.bfloat16,
        # Check whether your machine support this.
        # attn_implementation="flash_attention_2"
    ),
    client=dict(name='base_llm_client', client_num=1),
    server=dict(name='base_llm_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=1, logging_path='./logging/enron_email_central_gemma2_2b')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_hack_pipeline

    generic_model_hack_pipeline(exp_args, seed=0)
