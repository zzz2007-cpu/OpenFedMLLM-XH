import torch
from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, client_sampling, VariableMonitor, LRScheduler, get_launcher

from fling_llm.model import export_hf_model, add_wrapper
from fling_llm.utils import compile_config, smart_copy


def generic_model_pipeline(args: dict, seed: int = 0) -> None:
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

    # Construct logger.
    logger = Logger(args.other.logging_path)
    
    # Initialize metrics for summary
    import time
    start_time = time.time()
    best_metric = {'loss': float('inf'), 'round': -1}
    round_times = []
    comm_history = []
    
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
    print(model)
    model = add_wrapper(model, **peft_config)

    for i in range(args.client.client_num):
        group.append(get_client(args=args, model=smart_copy(model), client_id=i, train_dataset=train_sets[i]))
    group.initialize()

    # Setup lr_scheduler.
    lr_scheduler = LRScheduler(base_lr=args.learn.scheduler.base_lr, args=args.learn.scheduler)
    # Setup launcher.
    launcher = get_launcher(args)

    # Training loop
    for i in range(args.learn.global_eps):
        round_start = time.time()
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
        comm_history.append(trans_cost)
        
        # Track round time
        round_end = time.time()
        round_times.append(round_end - round_start)

        # Logging train variables.
        mean_train_variables = train_monitor.variable_mean()
        mean_train_variables.update({'trans_cost': trans_cost / 1e6, 'lr': cur_lr})
        logger.add_scalars_dict(prefix='train', dic=mean_train_variables, rnd=i)

        # Testing
        if i % args.other.test_freq == 0 and "after_aggregation" in args.learn.test_place:
            test_result = group.server.test(model=group.clients[0].model)
            
            # Update best metric
            if test_result.get('eval loss', float('inf')) < best_metric['loss']:
                best_metric['loss'] = test_result['eval loss']
                best_metric['round'] = i

            # Logging test variables.
            logger.add_scalars_dict(prefix='after_aggregation_test', dic=test_result, rnd=i)

            # Saving model checkpoints.
            # torch.save(group.server.glob_dict, os.path.join(args.other.logging_path, 'model.ckpt'))
        torch.cuda.empty_cache()

    # --- Summary Reporting ---
    import numpy as np
    import time
    
    # Calculate Total Training Time
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    # Calculate Peak GPU Memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    
    # Calculate Total Communication Cost (Estimate)
    # trans_cost in logger is per round, accumulated roughly
    total_comm_mb = sum(comm_history) / 1e6 if 'comm_history' in locals() else 0.0

    print("\n" + "="*60)
    print(f"EXPERIMENT SUMMARY | Date: {time.asctime()}")
    print("="*60)
    
    # 1. Model Performance
    print(f"► Model Performance:")
    final_test_loss = test_result.get('eval loss', None) if 'test_result' in locals() else None
    if final_test_loss:
        try:
            ppl = np.exp(final_test_loss)
            print(f"  • Final Test Loss:       {final_test_loss:.4f}")
            print(f"  • Final Perplexity (PPL): {ppl:.4f}")
        except OverflowError:
            print(f"  • Final Perplexity (PPL): INF")
            
    if best_metric['loss'] != float('inf'):
        try:
            best_ppl = np.exp(best_metric['loss'])
            print(f"  • Best Test Loss:        {best_metric['loss']:.4f} (Round {best_metric['round']})")
            print(f"  • Best Perplexity:       {best_ppl:.4f}")
        except:
            pass
            
    # 2. System Efficiency
    print(f"\n► System Efficiency:")
    print(f"  • Total Training Time:   {hours}h {minutes}m ({total_time:.2f}s)")
    if 'round_times' in locals() and len(round_times) > 0:
        avg_round_time = sum(round_times) / len(round_times)
        print(f"  • Avg Round Time:        {avg_round_time:.2f}s")
    print(f"  • Peak GPU Memory:       {peak_memory:.2f} GB")
    # print(f"  • Total Comm (Est.):     {total_comm_mb:.2f} MB")
    
    # 3. Experiment Config
    print(f"\n► Configuration:")
    print(f"  • Rounds: {args.learn.global_eps} | Clients: {args.client.client_num} | Sample Rate: {args.client.sample_rate}")
    print(f"  • Local Epochs: {args.learn.local_eps} | Batch Size: {args.learn.batch_size}")
    print("="*60 + "\n")

