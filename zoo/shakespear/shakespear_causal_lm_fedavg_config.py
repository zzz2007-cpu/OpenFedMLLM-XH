import platform
import os
from easydict import EasyDict

# Increase HuggingFace timeout to handle unstable network connections
os.environ['HF_HUB_ETAG_TIMEOUT'] = '100'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '100'

exp_args = dict(
    data=dict(
        dataset='shakespear',
        data_path='Trelis/tiny-shakespeare',
        tokenizer='openai-community/gpt2',
        hacker_tokenizer='openai-community/gpt2',
        max_len=512,
        sample_method=dict(name='uniform', train_num=0, test_num=0)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=5,  # 增加本地训练轮数，确保本地模型学到知识
        local_iters=-1,
        global_eps=50,  # 保持足够的全局通信轮次
        batch_size=8,  # 适当增大 batch size 以稳定梯度
        test_place=['after_aggregation'],
        trainer=dict(name='sft_fedavg_trainer'),
        scheduler=dict(
            name='fix',
            base_lr=1e-4  # 稍微调大 LR，加速收敛
        ),
        hf_args=dict()
    ),
    model=dict(
        model_path='openai-community/gpt2',
        pretrained=True,
        peft_config=dict(name=None)
    ),
    client=dict(name='base_llm_client', client_num=10, sample_rate=1, val_frac=0),  # 全量客户端参与，减少随机性波动
    server=dict(name='base_llm_server'),
    group=dict(
        name='base_group',
        aggregation_method='avg',
        aggregation_parameters=dict(name='all'),
        include_non_param=True
    ),
    launcher=dict(
        name='serial'
    ) if platform.system().lower() != 'linux' else dict(name='multiprocessing', num_proc=2),
    other=dict(
        test_freq=1,
        logging_path='./logging/shakespear_fedavg_gpt2',
        resume_path=None,
        print_config=False
    )
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)
