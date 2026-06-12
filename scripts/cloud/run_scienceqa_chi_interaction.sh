#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ONE="${ROOT_DIR}/scripts/cloud/run_one_scienceqa_chi.sh"

# FedAvg-LoRA 2x2: first establish whether label and modality heterogeneity interact.
CONFIGS=(
  scienceqa_qwen25vl_fedavg_lora_L0_M0
  scienceqa_qwen25vl_fedavg_lora_L1_M0
  scienceqa_qwen25vl_fedavg_lora_L0_M1
  scienceqa_qwen25vl_fedavg_lora_L1_M1
)

for cfg in "${CONFIGS[@]}"; do
  CONFIG_NAME="${cfg}" "${RUN_ONE}"
done
