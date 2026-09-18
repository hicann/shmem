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

set -eu
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
EXEC_BIN="${PROJECT_ROOT}/build/bin/rdma_atomic_perftest"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/lib64:${LD_LIBRARY_PATH:-}"
cd -- "${SCRIPT_DIR}"

matrix=false
args=()
explicit_pe=false
help=false
for arg in "$@"; do
    case "${arg}" in
        --all) matrix=true ;;
        --pe-id|--pe-id=*) explicit_pe=true; args+=("${arg}") ;;
        -h|--help) help=true; args+=("${arg}") ;;
        *) args+=("${arg}") ;;
    esac
done
if ${help}; then
    echo "run.sh [--all] [binary options]: --all runs 13 batch and 52 latency configurations."
    echo "An explicit --pe-id runs one PE for cross-node single-case launches."
fi
if [[ ! -x "${EXEC_BIN}" ]]; then
    echo "Build with -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE first." >&2
    exit 1
fi
if ${help}; then
    exec "${EXEC_BIN}" --help
fi
if ${matrix}; then
    for arg in "${args[@]}"; do
        case "${arg}" in
            --pe-id|--pe-id=*|--test|--test=*|--op|--op=*|--data-type|--data-type=*)
                echo "--all selects both PEs, tests, operations and types; do not override these options." >&2
                exit 1
                ;;
        esac
    done
elif ${explicit_pe}; then
    exec "${EXEC_BIN}" "${args[@]}"
fi

run_case() {
    "${EXEC_BIN}" "$@" --pe-id 0 &
    local pe0_pid=$!
    "${EXEC_BIN}" "$@" --pe-id 1 &
    local pe1_pid=$!
    local status=0
    wait "${pe0_pid}" || status=1
    wait "${pe1_pid}" || status=1
    return "${status}"
}

if ${matrix}; then
    operations=(inc fetch_inc set add fetch_add and fetch_and or fetch_or xor fetch_xor swap compare_swap)
    for op in "${operations[@]}"; do
        run_case "${args[@]}" --test batch --op "${op}" --data-type uint64
    done
    for op in "${operations[@]}"; do
        for data_type in uint32 uint64 int32 int64; do
            run_case "${args[@]}" --test latency --op "${op}" --data-type "${data_type}"
        done
    done
else
    run_case "${args[@]}"
fi
