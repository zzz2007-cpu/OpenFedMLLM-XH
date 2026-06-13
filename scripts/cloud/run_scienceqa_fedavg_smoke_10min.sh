#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ONE="${ROOT_DIR}/scripts/cloud/run_one_scienceqa_chi.sh"

export CONFIG_NAME="${CONFIG_NAME:-scienceqa_qwen25vl_fedavg_lora_L0_M0}"
export ROUNDS="${ROUNDS:-1}"
export NUM_CLIENTS="${NUM_CLIENTS:-2}"
export CLIENTS_PER_ROUND="${CLIENTS_PER_ROUND:-2}"
export MAX_STEPS="${MAX_STEPS:-1}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export EVAL_EVERY="${EVAL_EVERY:-1}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-2}"
export FINAL_EVAL_FULL="${FINAL_EVAL_FULL:-0}"
export SAVE_GLOBAL_MODEL_FREQ="${SAVE_GLOBAL_MODEL_FREQ:-1}"
export SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-9999}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
export PROCESSOR_MIN_PIXELS="${PROCESSOR_MIN_PIXELS:-3136}"
export PROCESSOR_MAX_PIXELS="${PROCESSOR_MAX_PIXELS:-200704}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/scienceqa_smoke_10min/${RUN_ID}}"
TIME_LIMIT="${TIME_LIMIT:-10m}"
SEED="${SEED:-42}"
export SEED

run_name="${CONFIG_NAME#scienceqa_qwen25vl_}"
run_dir="${OUTPUT_ROOT}/${run_name}/seed_${SEED}"
checkpoint_dir="${run_dir}/global_round_${ROUNDS}"

echo "[Smoke-10m] full-chain check: load -> train -> aggregate -> checkpoint -> eval"
echo "[Smoke-10m] config=${CONFIG_NAME}"
echo "[Smoke-10m] rounds=${ROUNDS} clients=${NUM_CLIENTS} clients_per_round=${CLIENTS_PER_ROUND} max_steps=${MAX_STEPS}"
echo "[Smoke-10m] eval_every=${EVAL_EVERY} eval_samples=${EVAL_MAX_SAMPLES} final_eval_full=${FINAL_EVAL_FULL}"
echo "[Smoke-10m] output_root=${OUTPUT_ROOT} time_limit=${TIME_LIMIT}"

start_seconds="$(date +%s)"
timeout --signal=TERM --kill-after=30s "${TIME_LIMIT}" "${RUN_ONE}"
elapsed_seconds="$(( $(date +%s) - start_seconds ))"

required_files=(
  "${run_dir}/config_resolved.json"
  "${run_dir}/train_log.jsonl"
  "${run_dir}/round_metrics.jsonl"
  "${run_dir}/eval_metrics_per_round.jsonl"
  "${run_dir}/eval_metrics.json"
  "${run_dir}/final_summary.json"
  "${run_dir}/eval/predictions.jsonl"
  "${checkpoint_dir}/adapter_config.json"
  "${checkpoint_dir}/adapter_model.safetensors"
  "${checkpoint_dir}/openfed_global_state.safetensors"
  "${checkpoint_dir}/openfed_global_checkpoint.json"
)

for path in "${required_files[@]}"; do
  test -s "${path}" || {
    echo "[Smoke-10m] ERROR: missing or empty artifact: ${path}" >&2
    exit 1
  }
done

python3 - "${run_dir}" "${ROUNDS}" "${NUM_CLIENTS}" "${CLIENTS_PER_ROUND}" \
  "${MAX_STEPS}" "${EVAL_MAX_SAMPLES}" "${FINAL_EVAL_FULL}" "${elapsed_seconds}" <<'PY'
import json
import sys
from pathlib import Path

(
    run_dir_raw,
    rounds_raw,
    num_clients_raw,
    clients_per_round_raw,
    max_steps_raw,
    eval_max_samples_raw,
    final_eval_full_raw,
    elapsed_raw,
) = sys.argv[1:]

run_dir = Path(run_dir_raw)
rounds = int(rounds_raw)
num_clients = int(num_clients_raw)
clients_per_round = int(clients_per_round_raw)
max_steps = int(max_steps_raw)
eval_max_samples = int(eval_max_samples_raw)
final_eval_full = final_eval_full_raw == "1"

config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
assert config["fed_args"]["num_rounds"] == rounds
assert config["fed_args"]["num_clients"] == num_clients
assert config["fed_args"]["sample_clients"] == clients_per_round
assert config["training_args"]["max_steps"] == max_steps
assert config["eval_args"]["max_samples"] == eval_max_samples
assert config["eval_args"]["final_eval_full"] is final_eval_full

round_records = [
    json.loads(line)
    for line in (run_dir / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(round_records) == rounds
assert len(round_records[-1]["clients_this_round"]) == clients_per_round

per_round_eval = [
    json.loads(line)
    for line in (run_dir / "eval_metrics_per_round.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert per_round_eval, "per-round evaluation did not run"
assert per_round_eval[-1]["num_samples"] == eval_max_samples

final_metrics = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
assert final_metrics["num_samples"] == eval_max_samples
final_summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))

print("[Smoke-10m] PASS")
print(f"[Smoke-10m] elapsed_seconds={elapsed_raw}")
print(f"[Smoke-10m] run_dir={run_dir}")
print(f"[Smoke-10m] clients_this_round={round_records[-1]['clients_this_round']}")
print(f"[Smoke-10m] final_accuracy={final_summary.get('final_accuracy')}")
print(f"[Smoke-10m] final_macro_f1={final_summary.get('final_macro_f1')}")
PY
