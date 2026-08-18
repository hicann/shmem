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

set -e

usage() {
    echo "Usage: $0 [-ip <ip>] [-port <port>]"
    exit 1
}

ip="127.0.0.1"
port="8899"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -ip)
            [[ -n "${2:-}" ]] || usage
            ip="$2"
            shift 2
            ;;
        -port)
            [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]] || usage
            port="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd ${script_dir}/../../ && pwd)"
export PROJECT_ROOT=${project_root}
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH

cd "${PROJECT_ROOT}"
pids=()
for pe_id in 0 1; do
    ./build/bin/rdma_aggregate_demo 2 "${pe_id}" "tcp://${ip}:${port}" 2 0 0 &
    pids+=("$!")
done

ret=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        ret=1
    fi
done
exit "${ret}"
