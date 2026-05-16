#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

PIP_TIMEOUT="${PIP_TIMEOUT:-300}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_ATTEMPTS_PER_INDEX="${PIP_ATTEMPTS_PER_INDEX:-2}"
PIP_PRIMARY_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

pip_install_with_fallback() {
    local -a pip_args=("$@")
    local -a indexes=(
        "${PIP_PRIMARY_INDEX}"
        "https://pypi.org/simple"
        "https://mirrors.aliyun.com/pypi/simple"
    )

    local index attempt
    for index in "${indexes[@]}"; do
        for ((attempt=1; attempt<=PIP_ATTEMPTS_PER_INDEX; attempt++)); do
            echo "[pip] index=${index}, attempt ${attempt}/${PIP_ATTEMPTS_PER_INDEX}"
            if "${PYTHON_BIN}" -m pip install \
                --default-timeout "${PIP_TIMEOUT}" \
                --retries "${PIP_RETRIES}" \
                --progress-bar off \
                -i "${index}" \
                "${pip_args[@]}"; then
                return 0
            fi
            echo "[pip] attempt failed, retrying or falling back..."
        done
    done

    echo "[pip] all mirrors failed"
    return 1
}

pip_install_with_fallback --no-deps -r requirements.txt

if [[ "${INSTALL_QWEN2VL:-0}" == "1" ]] && [[ -f requirements-qwen2vl.txt ]]; then
    pip_install_with_fallback --no-deps -r requirements-qwen2vl.txt
fi

pip_install_with_fallback --no-deps -e .
