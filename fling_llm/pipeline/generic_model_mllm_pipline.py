# @Time    : 2025/4/9 10:38
# @Author  : Guogang Zhu
# @File    : generic_model_mmlm_pipline.py
# @Software: PyCharm

import os
import torch
from easydict import EasyDict
from types import MethodType

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, client_sampling, VariableMonitor, LRScheduler, get_launcher

from fling_llm.model import export_hf_model, add_wrapper
from fling_llm.utils import compile_config, smart_copy

def generic_model_mllm_pipeline(args: dict, seed: int = 0) -> None:
    r"""
    Overview:
       Pipeline for generic federated learning. Under this setting, models of each client is the same.
       The final performance of this generic model is tested on the server (typically using a global test dataset).
    Arguments:
        - args: dict type arguments.
        - seed: random seed.
    """
    # Compile the input arguments first.
    args = compile_config(args, seed)
    for k, v in args.items():
        print(k, v)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Initialize the model using args.
    peft_config = args.model.pop('peft_config')
    # finetune_config = args.model.pop('finetune_config')

    model = export_hf_model(args.model)
    print("Model Type:", type(model))

    # Freeze parameters within vision model or llm model
    if not args.finetune_config.tune_vision:
        model.vpm.requires_grad_(False)
    if not args.finetune_config.tune_llm:
        model.llm.requires_grad_(False)

    for name, param in model.llm.named_parameters():
        param.requires_grad = False

    # Strict pure-LoRA mode: do not keep any full module in modules_to_save.
    peft_config.update({"modules_to_save": None})

    if not hasattr(model, 'get_input_embeddings'):
        def get_input_embeddings(self):
            return self.llm.get_input_embeddings()
        model.get_input_embeddings = MethodType(get_input_embeddings, model)

    model = add_wrapper(model, **peft_config)

    model.print_trainable_parameters()

    model.enable_input_require_grads()

    # Load dataset
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = args.data.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = args.data.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    print("Model Configure:", model.config)

    args.data.update({"batch_vision": batch_vision})

    train_sets = []

    for i in range(args.client.client_num):
        data_path = os.path.join(args.data.data_path, f"client_{i}.json")
        train_sets.append(get_dataset(args, slice_config=slice_config, train=True, data_path=data_path))

    import copy
    test_set = copy.deepcopy(train_sets[0])

    # Load dataset.
    # train_set = get_dataset(args, train=True)
    # test_set = get_dataset(args, train=False)
    # Split dataset into clients.
    # train_sets = data_sampling(train_set, args, seed, train=True)

    # Initialize group, clients and server.
    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=test_set)

    for i in range(args.client.client_num):
        group.append(get_client(args=args, model=smart_copy(model), client_id=i, train_dataset=train_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.scheduler.base_lr, args=args.learn.scheduler)
    # Setup launcher.
    launcher = get_launcher(args)

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
            test_result = group.server.test(model=group.clients[0].hacker_model)
            # Logging test variables.
            logger.add_scalars_dict(prefix='before_aggregation_test', dic=test_result, rnd=i)

        # Aggregate parameters in each client.
        trans_cost = group.aggregate(i)

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
            # torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))
        torch.cuda.empty_cache()
