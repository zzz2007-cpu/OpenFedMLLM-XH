import torch
from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, client_sampling, VariableMonitor, LRScheduler, get_launcher
from fling_llm.dataset.utils import get_union_dict_data

from fling_llm.model import export_hf_model, add_wrapper, export_hf_tokenizer
from fling_llm.utils import compile_config, smart_copy
from fling_llm.hacker.hacker_utils import exec_hacker_pipeline


def generic_model_hack_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for hacking generic federated learning. Under this setting, models of each client is the same.
       The final performance of this generic model is tested on the server (typically using a global test dataset).
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load dataset.
    train_set = get_dataset(args, train=True)
    test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    train_sets = data_sampling(train_set, args, seed, train=True)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)

    # Initialize the model using args.
    peft_config = args.model.pop('peft_config')
    model = export_hf_model(args.model)
    model = add_wrapper(model, **peft_config)
    ref_model = model

    for i in range(args.client.client_num):
        group.append(get_client(args=args, model=smart_copy(model), client_id=i, train_dataset=train_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.scheduler.base_lr, args=args.learn.scheduler)
    # Setup launcher.
    launcher = get_launcher(args)

    # initial hack
    ref_data = get_union_dict_data(
        [
            [group.clients[i].train_dataset.data[idx] for idx in range(group.clients[i].num_iters)]
            for i in range(len(group.clients))
        ]
    )
    # complete hack
    hack_res = exec_hacker_pipeline(
        cfg=args,
        ref_data=ref_data,
        model=group.clients[0].model,
        ref_model=ref_model,
        tokenizer=export_hf_tokenizer(args.data.tokenizer),
        decode_method="greedy",
        pipeline="complete"
    )
    logger.add_scalars_dict(prefix='hack', dic=hack_res, rnd=0)
    # most_similar hack
    hack_res = exec_hacker_pipeline(
        cfg=args,
        ref_data=ref_data,
        model=group.clients[0].model,
        ref_model=ref_model,
        tokenizer=export_hf_tokenizer(args.data.tokenizer),
        decode_method="greedy",
        pipeline="most_similar"
    )
    logger.add_scalars_dict(prefix='hack', dic=hack_res, rnd=0)

    # Training loop
    for i in range(args.learn.global_eps):
        logger.logging('Starting round: ' + str(i))
        # Initialize variable monitor.
        train_monitor = VariableMonitor()

        # Random sample participated clients in each communication round.
        participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)

        # Adjust learning rate.
        cur_lr = lr_scheduler.get_lr(train_round=i)

        # Local training for each participated client and add results to the monitor.
        # Use multiprocessing for acceleration.
        train_results = launcher.launch(
            clients=[group.clients[j] for j in participated_clients],
            lr=cur_lr,
            task_name='train',
            **args.learn.hf_args
        )
        for item in train_results:
            train_monitor.append(item)

        # Testing
        if i % args.other.test_freq == 0 and "before_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)
            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=test_result, rnd=i)

        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i)

        # Hack at here !
        ref_data = get_union_dict_data(
            [
                [group.clients[i].train_dataset.data[idx] for idx in group.clients[i].train_dataset.idxes]
                for i in range(len(group.clients))
            ]
        )
        # complete hack
        hack_res = exec_hacker_pipeline(
            cfg=args,
            ref_data=ref_data,
            model=group.clients[0].model,
            ref_model=ref_model,
            tokenizer=export_hf_tokenizer(args.data.tokenizer),
            pipeline="complete"
        )
        logger.add_scalars_dict(prefix='hack', dic=hack_res, rnd=i + 1)
        # most_similar hack
        hack_res = exec_hacker_pipeline(
            cfg=args,
            ref_data=ref_data,
            model=group.clients[0].model,
            ref_model=ref_model,
            tokenizer=export_hf_tokenizer(args.data.tokenizer),
            pipeline="most_similar"
        )
        logger.add_scalars_dict(prefix='hack', dic=hack_res, rnd=i + 1)

        ref_model = smart_copy(group.clients[0].model)

        # Logging train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost': trans_cost / 1e6, 'lr': cur_lr})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=test_result, rnd=i)

            # Saving model checkpoints.
            # if i == (args.learn.global_eps - 1):
            # torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))

        torch.cuda.empty_cache()
