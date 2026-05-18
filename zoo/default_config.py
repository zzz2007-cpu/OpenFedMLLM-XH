import platform

default_exp_args = dict(
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
        local_eps=1,
        local_iters=-1,
        global_eps=4,
        batch_size=32,
        test_place=['after_aggregation'],
        trainer=dict(name='sft_fedavg_trainer'),
        scheduler=dict(
            name='fix',
            base_lr=2e-5
        ),
        hf_args=dict()
    ),
    model=dict(
        model_path='openai-community/gpt2',
        pretrained=True,
        peft_config=dict(name=None)
    ),
    client=dict(
        name='base_llm_client',
        client_num=10,
        sample_rate=1,
        val_frac=0
    ),
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
        logging_path='./logging/default_llm_experiment',
        resume_path=None,
        print_config=False
    ),
)
