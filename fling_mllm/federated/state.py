import copy
import torch


def _alg_match(fed_alg, target):
    fed_alg = str(fed_alg).lower()
    return fed_alg == target or target in fed_alg


def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if _alg_match(fed_args.fed_alg, "fedadagrad") or _alg_match(fed_args.fed_alg, "fedyogi") or _alg_match(fed_args.fed_alg, "fedadam"):
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau ** 2
    elif _alg_match(fed_args.fed_alg, "fedavgm"):
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict


def get_auxiliary_dict(fed_args, global_dict):
    if _alg_match(fed_args.fed_alg, "scaffold"):
        global_auxiliary = {}
        for key in global_dict.keys():
            # Keep control variates in fp32 for numerical stability.
            global_auxiliary[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.num_clients)]
    else:
        global_auxiliary = None
        auxiliary_model_list = [None] * fed_args.num_clients
        auxiliary_delta_dict = [None] * fed_args.num_clients
    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict
