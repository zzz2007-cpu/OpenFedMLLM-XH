#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ONE="${ROOT_DIR}/scripts/cloud/run_one_scienceqa_chi.sh"

export ROUNDS="${ROUNDS:-15}"
export SAVE_GLOBAL_MODEL_FREQ="${SAVE_GLOBAL_MODEL_FREQ:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/scienceqa_chi_15round/${RUN_ID}}"

CONFIGS=(
  scienceqa_qwen25vl_fedavg_lora_L0_M0
  scienceqa_qwen25vl_fedavg_lora_L1_M0
  scienceqa_qwen25vl_fedavg_lora_L0_M1
  scienceqa_qwen25vl_fedavg_lora_L1_M1
)

echo "[FedAvg-15] output_root=${OUTPUT_ROOT}"

for cfg in "${CONFIGS[@]}"; do
  CONFIG_NAME="${cfg}" "${RUN_ONE}"

  run_name="${cfg#scienceqa_qwen25vl_}"
  checkpoint_dir="${OUTPUT_ROOT}/${run_name}/seed_${SEED:-42}/global_round_${ROUNDS}"
  test -s "${checkpoint_dir}/adapter_config.json"
  test -s "${checkpoint_dir}/openfed_global_state.safetensors"
  test -s "${checkpoint_dir}/adapter_model.safetensors"
  test -s "${checkpoint_dir}/openfed_global_checkpoint.json"
  echo "[FedAvg-15] verified final global checkpoint: ${checkpoint_dir}"
done
