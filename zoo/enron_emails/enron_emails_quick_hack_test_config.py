import torch
import os
from easydict import EasyDict

# Increase HuggingFace timeout
os.environ['HF_HUB_ETAG_TIMEOUT'] = '100'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '100'

# Configuration for Quick Hack Test
# We will use 'most_similar' as default here, but the code structure allows testing 'complete' as well.
# In generic_model_hack_pipeline, it actually runs BOTH pipelines sequentially or as configured.
# Let's configure a small setup to test the hack pipeline flow.

hack_pipeline = 'most_similar'  # ['most_similar', 'complete']
hack_decode_method = 'greedy'   # ['diff', 'greedy']

exp_args = dict(
    data=dict(
        dataset='enron-email',
        data_path='Trelis/tiny-shakespeare', # Use small public dataset for quick test
        # In real scenario, use: './data/enron-emails'
        tokenizer='openai-community/gpt2',   # Small robust tokenizer
        max_len=64,
        sample_method=dict(name='uniform', train_num=10, test_num=5)
    ),
    learn=dict(
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        local_eps=1,
        local_iters=2,
        global_eps=2,
        batch_size=2,
        trainer=dict(name='sft_fedavg_trainer'),
        hf_args=dict(
            gradient_accumulation_steps=1,
            report_to="none"
        )
    ),
    model=dict(
        model_path='openai-community/gpt2', # Small model for speed
        pretrained=True,
        peft_config=dict(name=None),
        trust_remote_code=True,
    ),
    hacker=dict(
        pipeline=hack_pipeline,
        decode_method=hack_decode_method,
        generate_config=dict(
            max_new_tokens=32, # Generate short text for speed
            do_sample=False,
        )
    ),
    client=dict(name='base_llm_client', client_num=2, sample_rate=1, val_frac=0),
    server=dict(name='base_llm_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(
        test_freq=1, 
        logging_path=f'./logging/enron_quick_hack_test_gpt2_{hack_decode_method}_{hack_pipeline}',
        print_config=False
    )
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_hack_pipeline
    generic_model_hack_pipeline(exp_args, seed=42)
