#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ONE="${ROOT_DIR}/scripts/cloud/run_one_scienceqa_chi.sh"

CONFIGS=(
  scienceqa_qwen25vl_fedavg_lora_L2_M2
  scienceqa_qwen25vl_fedprox_lora_L2_M2
  scienceqa_qwen25vl_fedchi_L2_M2
)

for cfg in "${CONFIGS[@]}"; do
  CONFIG_NAME="${cfg}" "${RUN_ONE}"
done

