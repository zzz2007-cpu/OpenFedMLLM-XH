#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${CONFIG_NAME:?CONFIG_NAME is required, e.g. scienceqa_qwen25vl_fedavg_lora_L0_M0}"
: "${DATA_ROOT:=${ROOT_DIR}/ScienceQA/image_present}"
: "${SCIENCEQA_FED_DIR:=${ROOT_DIR}/data/scienceqa/federated_chi}"
: "${MODEL_PATH:=${ROOT_DIR}/../Qwen2.5-VL-3B-Instruct-ms}"
: "${OUTPUT_ROOT:=${ROOT_DIR}/outputs/scienceqa_chi}"
: "${SEED:=42}"
: "${DRY_RUN:=0}"

export DATA_ROOT SCIENCEQA_FED_DIR MODEL_PATH OUTPUT_ROOT SEED

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_PATH="${OUTPUT_ROOT}/logs/${CONFIG_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
CMD=(python3 mllmzoo/run_experiment.py --name "${CONFIG_NAME}")

echo "[ScienceQA-CHI] root=${ROOT_DIR}"
echo "[ScienceQA-CHI] config=${CONFIG_NAME}"
echo "[ScienceQA-CHI] fed_dir=${SCIENCEQA_FED_DIR}"
echo "[ScienceQA-CHI] model=${MODEL_PATH}"
echo "[ScienceQA-CHI] output_root=${OUTPUT_ROOT}"
echo "[ScienceQA-CHI] seed=${SEED}"
echo "[ScienceQA-CHI] log=${LOG_PATH}"
echo "[ScienceQA-CHI] command=${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ScienceQA-CHI] ERROR: model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${SCIENCEQA_FED_DIR}" ]]; then
  echo "[ScienceQA-CHI] ERROR: federated data directory not found: ${SCIENCEQA_FED_DIR}" >&2
  exit 1
fi

set +e
"${CMD[@]}" >"${LOG_PATH}" 2>&1
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
  echo "[ScienceQA-CHI] ERROR: experiment failed with exit code ${status}" >&2
  echo "[ScienceQA-CHI] Last 80 log lines:" >&2
  tail -n 80 "${LOG_PATH}" >&2 || true
  exit "${status}"
fi

echo "[ScienceQA-CHI] completed: ${CONFIG_NAME}"
