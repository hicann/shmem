#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

ENGINE=mte
PE_SIZE=2
NPU_COUNT=2
FIRST_NPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -engine) ENGINE="$2"; shift 2 ;;
        -pes) PE_SIZE="$2"; shift 2 ;;
        -gnpus) NPU_COUNT="$2"; shift 2 ;;
        -fnpu) FIRST_NPU="$2"; shift 2 ;;
        -ipport) export IPPORT="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数：$1" >&2; exit 1 ;;
    esac
done

if [[ "${ENGINE}" != "mte" && "${ENGINE}" != "udma" ]]; then
    echo "[ERROR] -engine 只接受 mte 或 udma。" >&2
    exit 1
fi
if [[ "${PE_SIZE}" -lt 2 || "${NPU_COUNT}" -lt "${PE_SIZE}" ]]; then
    echo "[ERROR] 示例要求至少 2 个 PE，且 NPU_COUNT 不能小于 PE_SIZE。" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ -n "${CANN_SET_ENV:-}" && -f "${CANN_SET_ENV}" ]]; then
        set +u
        source "${CANN_SET_ENV}"
        set -u
    else
        echo "[ERROR] 请先 source CANN set_env.sh。" >&2
        exit 1
    fi
fi
if [[ ! -f "${PROJECT_ROOT}/install/set_env.sh" ]]; then
    echo "[ERROR] 请先执行 bash scripts/build.sh -examples。" >&2
    exit 1
fi
set +u
source "${PROJECT_ROOT}/install/set_env.sh"
set -u
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export IPPORT="${IPPORT:-tcp://127.0.0.1:$((27010 + RANDOM % 900))}"
export SHMEM_UID_SESSION_ID="${SHMEM_UID_SESSION_ID:-127.0.0.1:$((8899 + RANDOM % 900))}"

EXE="${PROJECT_ROOT}/build/bin/user_buffer"
if [[ ! -x "${EXE}" ]]; then
    echo "[ERROR] ${EXE} 不存在，请先构建 examples。" >&2
    exit 1
fi

pids=()
terminate_children() {
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
}

handle_signal() {
    terminate_children
    exit 130
}

trap handle_signal INT TERM

for ((pe = 0; pe < PE_SIZE; ++pe)); do
    "${EXE}" "${PE_SIZE}" "${pe}" "${IPPORT}" "${NPU_COUNT}" "${FIRST_NPU}" "${ENGINE}" &
    pids+=("$!")
done

result=0
remaining=${#pids[@]}
while ((remaining > 0)); do
    wait -n
    status=$?
    ((remaining -= 1))
    if ((status != 0)); then
        result=${status}
        terminate_children
        break
    fi
done

trap - INT TERM
exit "${result}"
