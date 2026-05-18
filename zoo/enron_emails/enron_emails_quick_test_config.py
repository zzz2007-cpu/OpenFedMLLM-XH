import torch
import os
from easydict import EasyDict

# Increase HuggingFace timeout
os.environ['HF_HUB_ETAG_TIMEOUT'] = '100'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '100'

exp_args = dict(
    data=dict(
        dataset='enron-email',
        data_path='Trelis/tiny-shakespeare',  # Temporary use small dataset for test or ensure enron exists
        # In real case, if you have local enron data, use it. 
        # For quick test, we can use a small HF dataset or assume enron is downloaded.
        # Here we assume enron is available or use a public tiny one if needed.
        # Let's keep 'enron-email' but use a tiny subset logic if possible.
        # Since dataset code might need local file, we trust your environment has it or we use a public one.
        # To be safe for "quick test", let's use the robust GPT2 tokenizer and a small model.
        tokenizer='openai-community/gpt2',
        max_len=64, # Short sequence for speed
        sample_method=dict(name='uniform', train_num=10, test_num=5) # Only use 10 samples for training
    ),
    learn=dict(
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        local_eps=1,      # Only 1 local epoch
        local_iters=2,    # Or just 2 iterations per round
        global_eps=2,     # Only 2 global rounds
        batch_size=2,     # Small batch
        test_place=['after_aggregation'],
        trainer=dict(name='sft_fedavg_trainer'),
        scheduler=dict(
            name='fix',
            base_lr=1e-4
        ),
        hf_args=dict(
            # Disable heavy optimizations for quick debug on small machine
            gradient_accumulation_steps=1,
            report_to="none" 
        )
    ),
    model=dict(
        model_path='openai-community/gpt2', # Use GPT2-small (124M) instead of 7B/2B for instant test
        pretrained=True,
        peft_config=dict(name=None), # No LoRA for fastest raw test, or use dict(name='lora') if you want to test loRA
        trust_remote_code=True,
    ),
    client=dict(name='base_llm_client', client_num=2, sample_rate=1, val_frac=0), # Only 2 clients
    server=dict(name='base_llm_server'),
    group=dict(
        name='base_group', 
        aggregation_method='avg',
        aggregation_parameters=dict(name='all'),
        include_non_param=True
    ),
    other=dict(
        test_freq=1, 
        logging_path='./logging/enron_quick_test_gpt2',
        print_config=False
    )
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=42)
