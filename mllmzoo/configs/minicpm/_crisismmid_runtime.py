import os
from pathlib import Path


def resolve_runtime_overrides(
    *,
    default_data_path,
    default_eval_path,
    default_output_dir,
    default_fed_alg,
    default_num_rounds=20,
    default_num_clients=10,
):
    explicit_data_path = os.environ.get("OPENFED_CRISISMMID_DATA_PATH")
    alpha = os.environ.get("OPENFED_CRISISMMID_ALPHA")
    data_root = os.environ.get("OPENFED_CRISISMMID_DATA_ROOT")
    num_clients = int(os.environ.get("OPENFED_CRISISMMID_NUM_CLIENTS", str(default_num_clients)))

    if explicit_data_path:
        data_path = explicit_data_path
    elif alpha:
        if not data_root:
            data_root = str(Path(default_data_path).parent)
        data_path = os.path.join(data_root, f"partition-alpha{alpha}-clt{num_clients}")
    else:
        data_path = default_data_path

    eval_data_path = os.environ.get("OPENFED_CRISISMMID_EVAL_PATH", default_eval_path)
    output_dir = os.environ.get("OPENFED_CRISISMMID_OUTPUT_DIR", default_output_dir)
    fed_alg = os.environ.get("OPENFED_CRISISMMID_FED_ALG", default_fed_alg)
    num_rounds = int(os.environ.get("OPENFED_CRISISMMID_NUM_ROUNDS", str(default_num_rounds)))

    return {
        "data_path": data_path,
        "eval_data_path": eval_data_path,
        "output_dir": output_dir,
        "fed_alg": fed_alg,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "alpha": alpha,
    }
