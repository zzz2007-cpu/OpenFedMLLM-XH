import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from easydict import EasyDict

from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

exp_args = dict(
    data=dict(
        dataset='crisis-mmd',
        # Default alpaca: 'tatsu-lab/alpaca'
        # Alpaca-GPT4: 'vicgalle/alpaca-gpt4'
        data_path='/home/zgg/data/FedMLLM_Data/crisis-mmd/minicpmv_data/partition-alpha0.5-clt10',
        tokenizer='openbmb/MiniCPM-V-2_6-int4',
        max_slice_nums=4,
        vision_batch_size=8,
        max_len=1200,
        patch_size=14,
        query_nums=64,
        sample_method=dict(name='uniform', train_num=0, test_num=0)
    ),

    learn=dict(
        device='cuda:0',
        local_iters=500,
        global_eps=40,
        batch_size=1,
        use_lora=True,
        trainer=dict(name='mllm_sft_fedavg_trainer'),
        hf_args=dict(
            bf16=True,
            bf16_full_eval=True,
            fp16=False,
            fp16_full_eval=False,
            # tf32=True,
            disable_tqdm=False,
            remove_unused_columns=False,
            label_names='labels',
            prediction_loss_only=False,
            gradient_accumulation_steps=4,
            logging_steps=1 \
        )
    ),
    model=dict(
        model_path='openbmb/MiniCPM-V-2_6-int4',
        pretrained=True,
        trust_remote_code=True,
        revision='main',
        torch_dtype=torch.bfloat16,
        # Check whether your machine support this.
        # attn_implementation="flash_attention_2"
        peft_config=dict(name='lora', q_lora=False, r=8, lora_alpha=8, lora_dropout=0.05,
                         target_modules='llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',
                         bias='none', layers_to_transform=None),
    ),
    finetune_config=dict(
        llm_type='qwen2',
        tune_vision=False,
        tune_llm=False
    ),
    client=dict(name='base_llm_client', client_num=1),
    server=dict(name='base_llm_server'),
    group=dict(name='base_group', aggregation_method='avg', include_non_param=False),
    other=dict(test_freq=1, logging_path='./logging/crisis_fedavg_minicpm_v_2_6_int4')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_mllm_pipeline
    generic_model_mllm_pipeline(exp_args, seed=0)
