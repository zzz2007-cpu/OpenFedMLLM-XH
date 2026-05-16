#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export OPENFED_HATEFUL_MEMES_ROOT="${OPENFED_HATEFUL_MEMES_ROOT:-./hateful_memes}"
export OPENFED_HATEFUL_MEMES_STAT="${OPENFED_HATEFUL_MEMES_STAT:-${stat_setting:-dir_0.1}}"
export OPENFED_HATEFUL_MEMES_MODAL="${OPENFED_HATEFUL_MEMES_MODAL:-${modal_setting:-aligned}}"
export OPENFED_NUM_CLIENTS="${OPENFED_NUM_CLIENTS:-${num_clients:-10}}"
export OPENFED_SAMPLE_CLIENTS="${OPENFED_SAMPLE_CLIENTS:-${client_sample_num:-5}}"
export OPENFED_NUM_ROUNDS="${OPENFED_NUM_ROUNDS:-${rounds:-20}}"
export OPENFED_LOCAL_EPOCHS="${OPENFED_LOCAL_EPOCHS:-${local_epochs:-1}}"
export OPENFED_BATCH_SIZE="${OPENFED_BATCH_SIZE:-${batch_size:-1}}"
export OPENFED_LORA_R="${OPENFED_LORA_R:-${lora_r:-8}}"
export OPENFED_LORA_ALPHA="${OPENFED_LORA_ALPHA:-${lora_alpha:-16}}"
export OPENFED_LORA_DROPOUT="${OPENFED_LORA_DROPOUT:-${lora_dropout:-0.05}}"

python3 mllmzoo/run_experiment.py --name minicpm_v_2_6_int4_hateful_memes_fedavg "$@"
