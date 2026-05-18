import torch
from easydict import EasyDict


hack_decode_method = 'diff'  # ['diff', 'greedy']
exp_args = dict(
    data=dict(
        dataset='enron-email',
        data_path='./data/enron-emails',
        tokenizer='./data/Llama-2-7b-hf/',
        # tokenizer='meta-llama/Llama-2-7b-hf',
        max_len=1024,
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
        peft_config=dict(name=None),
        # Check whether your machine support this.
        # attn_implementation="flash_attention_2"
    ),
    hacker=dict(
        decode_method=hack_decode_method,
        generate_config=dict(
            max_new_tokens=256,
            # temperature=0.95,
            do_sample=False,
            # top_p=0.6,
            # repetition_penalty=1.2,
        )
    ),
    client=dict(name='base_llm_client', client_num=4),
    server=dict(name='base_llm_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=1, logging_path=f'./logging/enron_email_fedavg_llama2_7b_{hack_decode_method}_test')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_hack_pipeline

    generic_model_hack_pipeline(exp_args, seed=0)
