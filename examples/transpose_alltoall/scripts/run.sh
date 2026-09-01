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

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

USER_B=8
USER_N=32
USER_S=1024
USER_D=1024

VERBOSE=""
TEST_TYPE="float16_t"
SKIP_CHECK=0
DEVICE_LIST_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE="--verbose"; shift ;;
        -type|--dtype) TEST_TYPE="$2"; shift 2 ;;
        -nc|--skip-check|--no-check) SKIP_CHECK=1; shift ;;
        *) DEVICE_LIST_ARG="$1"; shift ;;
    esac
done
if [[ -z "${DEVICE_LIST_ARG}" ]]; then
    DEVICE_ID_LIST=(0)
else
    IFS=',' read -ra DEVICE_ID_LIST <<< "${DEVICE_LIST_ARG}"
fi
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if (( RANK_SIZE < 1 )); then
    echo "[ERROR] Need at least 1 device (got device list: '${DEVICE_LIST_ARG}')" >&2
    exit 1
fi

if (( USER_B % RANK_SIZE != 0 )); then
    echo "[ERROR] USER_B=${USER_B} is not divisible by RANK_SIZE=${RANK_SIZE}." >&2
    echo "        B-axis AllToAll splits B into R chunks per rank," >&2
    echo "        so B mod R must be 0." >&2
    echo "        Fix #1: reduce RANK_SIZE  (e.g. pass \"0\" instead of \"0,1\")" >&2
    echo "        Fix #2: edit USER_B in run.sh to a multiple of R" >&2
    exit 1
fi

echo "Devices     = [${DEVICE_ID_LIST[*]}], rankSize=${RANK_SIZE}"
echo "Shape       : B=${USER_B}, N=${USER_N}, S=${USER_S}, D=${USER_D}"
echo "Data type   : ${TEST_TYPE}"
if (( SKIP_CHECK == 1 )); then
    echo "Check       : skipped (--skip-check)"
else
    echo "Check       : enabled"
fi

cd ${PROJECT_ROOT}/examples/transpose_alltoall/
DATA_DIR=`realpath ./out`
mkdir -p "${DATA_DIR}"
echo "DATA_DIR: $DATA_DIR"
EXEC_BIN=${PROJECT_ROOT}/build/bin/transpose_alltoall
set -euo pipefail
IFS=$'\n\t'

ERROR_FLAG="/tmp/verify_failed_$$_$(date +%s)"
rm -f "$ERROR_FLAG"
BASE_PORT=$((20000 + ($$ % 10000)))

echo "Processing fixed user-specified case: B=${USER_B}, N=${USER_N}, S=${USER_S}, D=${USER_D}, rankSize=${RANK_SIZE}, dtype=${TEST_TYPE}"

rm -f ./out/*_output.bin 2>/dev/null || true
python3 "${SCRIPT_DIR}/gen_data.py" \
    --rankSize "${RANK_SIZE}" --B "${USER_B}" --N "${USER_N}" --S "${USER_S}" --D "${USER_D}" \
    --dtype "${TEST_TYPE}" \
    --data_dir "${DATA_DIR}"

IPPORT="tcp://127.0.0.1:${BASE_PORT}"
echo "Using SHMEM IPPORT: ${IPPORT}"

export SHMEM_UID_SESSION_ID=127.0.0.1:8899
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH:-}/lib64:${LD_LIBRARY_PATH:-}

pids=()
for (( idx = 0; idx < RANK_SIZE; idx = idx + 1 )); do
    (
        set +e
        if [[ -n "${DEVICE_LIST_ARG}" ]]; then
            ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$USER_B" "$USER_S" "$USER_N" "$USER_D" "$TEST_TYPE" "${DATA_DIR}" "${DEVICE_LIST_ARG}"
        else
            ${EXEC_BIN} "$RANK_SIZE" "$idx" "$IPPORT" "$USER_B" "$USER_S" "$USER_N" "$USER_D" "$TEST_TYPE" "${DATA_DIR}"
        fi
        rc=$?
        if (( rc != 0 )); then
            echo "[ERROR] rank=${idx} exited with status=${rc}" >&2
            echo "${rc}" > "${ERROR_FLAG}.rank${idx}"
        fi
        exit "${rc}"
    ) &
    pids+=($!)
done

exec_ok=1
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        exec_ok=0
    fi
done

for f in "${ERROR_FLAG}".rank*; do
    if [[ -e "$f" ]]; then
        exec_ok=0
        rm -f "$f"
    fi
done
if (( exec_ok != 1 )); then
    echo "[ERROR] One or more rank processes failed (rankSize=${RANK_SIZE})." >&2
    rm -f "${ERROR_FLAG}" "${ERROR_FLAG}".rank*
    cd "${CURRENT_DIR}"
    exit 1
fi

sleep 1

if (( SKIP_CHECK == 1 )); then
    echo "Skipping verification (--skip-check)."
else
    echo "Verifying results (standalone verify_results.py, 4-D tensor dump)..."
    echo "Per-rank input shape:  [B=${USER_B}, N=${USER_N}, S=${USER_S}, D=${USER_D}]"
    echo "Per-rank output shape: [B=${USER_B}, S=${USER_S}, N=${USER_N}, D=${USER_D}]"

    python3 "${SCRIPT_DIR}/verify_results.py" \
        --B "${USER_B}" --N "${USER_N}" --S "${USER_S}" --D "${USER_D}" \
        --rankSize "${RANK_SIZE}" --dtype "${TEST_TYPE}" \
        --input-pattern "./out/rank_{}_input.bin" \
        --golden-pattern "./out/rank_{}_golden.bin" \
        --actual-pattern "./out/rank_{}_output.bin" ${VERBOSE} 2>&1 || {
        echo "[ERROR] Verification failed" >&2
        touch "$ERROR_FLAG"
    }
fi

if [ -f "$ERROR_FLAG" ]; then
    echo "error case!"
    rm -f "$ERROR_FLAG"
    cd "${CURRENT_DIR}"
    exit 1
fi

echo "User-specified case (B=${USER_B},N=${USER_N},S=${USER_S},D=${USER_D},R=${RANK_SIZE},dtype=${TEST_TYPE}) passed successfully!"

cd ${CURRENT_DIR}
