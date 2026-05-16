import argparse
import os
import runpy
from pathlib import Path


CONFIG_BY_ALGO = {
    "fedavg": "minicpmv-crisismmid-FedAvg.py",
    "fedprox": "minicpmv-crisismmid-FedProx.py",
    "fednova": "minicpmv-crisismmid-FedNova.py",
    "scaffold": "minicpmv-crisismmid-Scaffold.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MiniCPM-V CrisisMMD federated training with runtime-selectable algorithm and alpha."
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(CONFIG_BY_ALGO),
        required=True,
        help="Federated algorithm to run.",
    )
    parser.add_argument(
        "--alpha",
        required=True,
        help="Dirichlet alpha used in partition directory name, e.g. 0.1, 0.5, 1.0.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Client count used in the partition directory name.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=20,
        help="Number of federated communication rounds.",
    )
    parser.add_argument(
        "--data-root",
        default="./data/crisis-mmd/minicpmv_data",
        help="Root directory that contains partition-alpha*-clt* folders.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional explicit partition path. Overrides --alpha/--data-root.",
    )
    parser.add_argument(
        "--eval-data-path",
        default="./data/crisis-mmd/minicpmv_data/test.json",
        help="Evaluation JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without launching training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / CONFIG_BY_ALGO[args.algorithm]

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(
            args.data_root,
            f"partition-alpha{args.alpha}-clt{args.num_clients}",
        )

    output_dir = args.output_dir or (
        f"./mllmzoo/output/minicpmv_crisismmid_{args.algorithm}_alpha{args.alpha}"
    )

    os.environ["OPENFED_CRISISMMID_ALPHA"] = args.alpha
    os.environ["OPENFED_CRISISMMID_NUM_CLIENTS"] = str(args.num_clients)
    os.environ["OPENFED_CRISISMMID_NUM_ROUNDS"] = str(args.num_rounds)
    os.environ["OPENFED_CRISISMMID_DATA_ROOT"] = args.data_root
    os.environ["OPENFED_CRISISMMID_DATA_PATH"] = data_path
    os.environ["OPENFED_CRISISMMID_EVAL_PATH"] = args.eval_data_path
    os.environ["OPENFED_CRISISMMID_OUTPUT_DIR"] = output_dir
    os.environ["OPENFED_CRISISMMID_FED_ALG"] = args.algorithm

    print("[minicpmv_crisismmid_runner] resolved configuration")
    print(f"  algorithm:      {args.algorithm}")
    print(f"  alpha:          {args.alpha}")
    print(f"  num_clients:    {args.num_clients}")
    print(f"  num_rounds:     {args.num_rounds}")
    print(f"  config:         {config_path}")
    print(f"  data_path:      {data_path}")
    print(f"  eval_data_path: {args.eval_data_path}")
    print(f"  output_dir:     {output_dir}")

    if args.dry_run:
        return

    runpy.run_path(str(config_path), run_name="__main__")


if __name__ == "__main__":
    main()
