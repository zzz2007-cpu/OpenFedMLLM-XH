import json
import os
import random
from numbers import Number
import torch


def _alg_match(fed_alg, target):
    fed_alg = str(fed_alg).lower()
    return fed_alg == target or target in fed_alg


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_fednova_tau_entry(entry, client_idx, eps):
    """
    Normalize client tau entry to float.
    Supports:
      - float/int
      - scalar tensor / numpy scalar (via .item())
      - dict stats (preferred): {"tau_eff": ..., "normalizer": ..., "local_steps": ...}
    """
    source = "default"
    value = 1.0
    raw = entry

    if raw is None:
        return max(float(value), float(eps)), source

    # torch scalar / numpy scalar
    if hasattr(raw, "item") and callable(getattr(raw, "item")):
        try:
            raw = raw.item()
        except Exception:
            pass

    if isinstance(raw, Number):
        value = float(raw)
        source = type(raw).__name__
        return max(value, float(eps)), source

    if isinstance(raw, dict):
        for key in ("tau_eff", "normalizer", "tau", "local_steps"):
            if key not in raw:
                continue
            cand = raw.get(key)
            if hasattr(cand, "item") and callable(getattr(cand, "item")):
                try:
                    cand = cand.item()
                except Exception:
                    pass
            if isinstance(cand, Number):
                value = float(cand)
                source = f"dict.{key}"
                return max(value, float(eps)), source
        raise TypeError(
            f"FedNova tau entry for client={client_idx} is dict but has no numeric "
            f"tau fields. keys={list(raw.keys())}"
        )

    raise TypeError(
        "FedNova tau entry type is unsupported: "
        f"client={client_idx}, type={type(entry).__name__}, value={repr(entry)[:200]}"
    )


def get_clients_this_round(fed_args, round_idx):
    if str(fed_args.fed_alg).startswith("local"):
        return [int(str(fed_args.fed_alg)[-1])]
    if fed_args.num_clients < fed_args.sample_clients:
        return list(range(fed_args.num_clients))
    random.seed(round_idx)
    return sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))


def global_aggregate(
    fed_args,
    global_dict,
    local_dict_list,
    sample_num_list,
    clients_this_round,
    round_idx,
    proxy_dict=None,
    opt_proxy_dict=None,
    auxiliary_info=None,
    fednova_info=None,
):
    sample_this_round = sum(sample_num_list[client] for client in clients_this_round)
    global_auxiliary = None
    if _alg_match(fed_args.fed_alg, "scaffold"):
        debug_scaffold = _env_flag("OPENFED_DEBUG_SCAFFOLD", default=False)
        delta_w_sq = 0.0
        for key in global_dict.keys():
            delta_w = sum(
                (local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            )
            if debug_scaffold:
                delta_w_sq += float(torch.sum(delta_w.float() * delta_w.float()).item())
            global_dict[key] = global_dict[key] + fed_args.scaffold_server_lr * delta_w
        global_auxiliary, auxiliary_delta_dict = auxiliary_info
        if global_auxiliary is not None and auxiliary_delta_dict is not None:
            delta_c_sq = 0.0
            for key in global_auxiliary.keys():
                delta_auxiliary = sum(auxiliary_delta_dict[client][key].float() for client in clients_this_round)
                delta_auxiliary = delta_auxiliary / fed_args.num_clients
                updated = global_auxiliary[key].float() + delta_auxiliary
                global_auxiliary[key] = updated
                if debug_scaffold:
                    delta_c_sq += float(torch.sum(delta_auxiliary * delta_auxiliary).item())
            if debug_scaffold:
                global_c_sq = 0.0
                for value in global_auxiliary.values():
                    global_c_sq += float(torch.sum(value.float() * value.float()).item())
                print(
                    "[SCAFFOLD][Server] "
                    + json.dumps(
                        {
                            "round_idx": int(round_idx),
                            "clients_this_round": [int(c) for c in clients_this_round],
                            "delta_w_l2": float(delta_w_sq**0.5),
                            "delta_c_l2": float(delta_c_sq**0.5),
                            "global_c_l2": float(global_c_sq**0.5),
                            "server_lr": float(getattr(fed_args, "scaffold_server_lr", 1.0)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    elif _alg_match(fed_args.fed_alg, "fedavgm"):
        for key in global_dict.keys():
            delta_w = sum((local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            global_dict[key] = global_dict[key] + proxy_dict[key]
    elif _alg_match(fed_args.fed_alg, "fedadagrad"):
        for key, param in opt_proxy_dict.items():
            delta_w = sum((local_dict_list[client][key] - global_dict[key]) for client in clients_this_round) / len(clients_this_round)
            proxy_dict[key] = delta_w
            opt_proxy_dict[key] = param + proxy_dict[key].square()
            global_dict[key] += fed_args.fedopt_eta * proxy_dict[key] / (opt_proxy_dict[key].sqrt() + fed_args.fedopt_tau)
    elif _alg_match(fed_args.fed_alg, "fedyogi"):
        for key, param in opt_proxy_dict.items():
            delta_w = sum((local_dict_list[client][key] - global_dict[key]) for client in clients_this_round) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            delta_square = proxy_dict[key].square()
            opt_proxy_dict[key] = param - (1 - fed_args.fedopt_beta2) * delta_square * (param - delta_square).sign()
            global_dict[key] += fed_args.fedopt_eta * proxy_dict[key] / (opt_proxy_dict[key].sqrt() + fed_args.fedopt_tau)
    elif _alg_match(fed_args.fed_alg, "fedadam"):
        for key, param in opt_proxy_dict.items():
            delta_w = sum((local_dict_list[client][key] - global_dict[key]) for client in clients_this_round) / len(clients_this_round)
            proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
            opt_proxy_dict[key] = fed_args.fedopt_beta2 * param + (1 - fed_args.fedopt_beta2) * proxy_dict[key].square()
            global_dict[key] += fed_args.fedopt_eta * proxy_dict[key] / (opt_proxy_dict[key].sqrt() + fed_args.fedopt_tau)
    elif _alg_match(fed_args.fed_alg, "fednova"):
        tau_list = fednova_info[0] if fednova_info is not None else None
        client_weight = {}
        client_tau = {}
        tau_eff = 0.0
        fednova_eps = float(getattr(fed_args, "fednova_eps", 1e-12))
        debug_enabled = _env_flag("OPENFED_DEBUG_FEDNOVA", default=True)
        debug_rows = []
        for client in clients_this_round:
            weight = sample_num_list[client] / sample_this_round
            tau_entry = None
            if tau_list is not None:
                if client >= len(tau_list):
                    raise IndexError(
                        f"FedNova tau_list index out of range: client={client}, len={len(tau_list)}"
                    )
                tau_entry = tau_list[client]
            tau, tau_source = _resolve_fednova_tau_entry(
                entry=tau_entry,
                client_idx=client,
                eps=fednova_eps,
            )
            client_weight[client] = weight
            client_tau[client] = tau
            tau_eff += weight * tau
            if debug_enabled:
                debug_rows.append({
                    "client": int(client),
                    "weight": float(weight),
                    "tau": float(tau),
                    "tau_source": tau_source,
                    "tau_entry_type": type(tau_entry).__name__ if tau_entry is not None else "NoneType",
                })
        if debug_enabled:
            print(
                "[FedNova][TauDebug] "
                + json.dumps(
                    {
                        "round_idx": int(round_idx),
                        "clients_this_round": [int(c) for c in clients_this_round],
                        "tau_eff": float(tau_eff),
                        "rows": debug_rows,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for key in global_dict.keys():
            normalized_delta = sum(
                (local_dict_list[client][key] - global_dict[key]) * client_weight[client] / client_tau[client]
                for client in clients_this_round
            )
            global_dict[key] = global_dict[key] + fed_args.fednova_server_lr * tau_eff * normalized_delta
    else:
        for key in global_dict.keys():
            value = None
            for client in clients_this_round:
                weight = sample_num_list[client] / sample_this_round
                value = local_dict_list[client][key] * weight if value is None else value + local_dict_list[client][key] * weight
            global_dict[key] = value
    return global_dict, global_auxiliary
