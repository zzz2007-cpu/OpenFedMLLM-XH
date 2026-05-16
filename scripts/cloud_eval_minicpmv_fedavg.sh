#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer explicit project root when provided; otherwise try cloud-defaults seen in your setup.
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -d "$ROOT_DIR/mllmzoo" ]]; then
  PROJECT_ROOT="$ROOT_DIR"
elif [[ -d "/home/zhangzhuangzhi/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/home/zhangzhuangzhi/OpenFedMLLM-master"
elif [[ -d "/root/zhangzhuangzhi/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/root/zhangzhuangzhi/OpenFedMLLM-master"
elif [[ -d "/workspace/zhangzhuangzhi/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/workspace/zhangzhuangzhi/OpenFedMLLM-master"
elif [[ -d "/data/zhangzhuangzhi/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/data/zhangzhuangzhi/OpenFedMLLM-master"
elif [[ -d "/mnt/zhangzhuangzhi/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/mnt/zhangzhuangzhi/OpenFedMLLM-master"
elif [[ -d "/mnt/data/zhangzhuangzhi/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/mnt/data/zhangzhuangzhi/OpenFedMLLM-master"
elif [[ -d "/home/ubuntu/OpenFedMLLM-master/mllmzoo" ]]; then
  PROJECT_ROOT="/home/ubuntu/OpenFedMLLM-master"
else
  PROJECT_ROOT="$ROOT_DIR"
fi

# Defaults (override via env vars)
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/mllmzoo/output/minicpmv_crisismmid_fedavg}"
TEST_DATA_PATH="${TEST_DATA_PATH:-$PROJECT_ROOT/data/crisis-mmd/minicpmv_data/test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/eval_full}"
BASE_MODEL="${BASE_MODEL:-openbmb/MiniCPM-V-2_6-int4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  # Pick the latest checkpoint-N by numeric suffix.
  latest_ckpt_num="$(ls -1d "$RUN_DIR"/checkpoint-* 2>/dev/null \
    | sed -E 's/.*checkpoint-([0-9]+)$/\1/' \
    | sort -n \
    | tail -n 1)"
  if [[ -z "${latest_ckpt_num}" ]]; then
    echo "[cloud-eval] ERROR: no checkpoint-* found under: $RUN_DIR" >&2
    exit 1
  fi
  CHECKPOINT_DIR="$RUN_DIR/checkpoint-$latest_ckpt_num"
fi

if [[ ! -f "$TEST_DATA_PATH" ]]; then
  echo "[cloud-eval] ERROR: test data not found: $TEST_DATA_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[cloud-eval] Using checkpoint: $CHECKPOINT_DIR"
echo "[cloud-eval] Using test data:  $TEST_DATA_PATH"
echo "[cloud-eval] Output dir:       $OUTPUT_DIR"

CMD=(
  python "$ROOT_DIR/fling_mllm/eval_fed_model.py"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --test_data_path "$TEST_DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --model_name_or_path "$BASE_MODEL"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
)

if [[ -n "${CACHE_DIR:-}" ]]; then
  CMD+=(--cache_dir "$CACHE_DIR")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  CMD+=(--max_samples "$MAX_SAMPLES")
fi

echo "[cloud-eval] Running: ${CMD[*]}"
exec "${CMD[@]}"
