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

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
PERF_BYTES=${PERF_BYTES:-8388608}
PERF_ITERATIONS=${PERF_ITERATIONS:-100}
PERF_WARMUP=${PERF_WARMUP:-10}
PERF_ROUNDS=${PERF_ROUNDS:-7}
PERF_THRESHOLD_PERCENT=${PERF_THRESHOLD_PERCENT:-5}
PERF_OUTPUT_DIR=${PERF_OUTPUT_DIR:-"${SCRIPT_DIR}/results"}

mkdir -p "${PERF_OUTPUT_DIR}"
bash "${SCRIPT_DIR}/build_cpp_ref.sh"

torchrun \
    --nproc-per-node "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/benchmark_putmem_on_stream.py" \
    --bytes "${PERF_BYTES}" \
    --iterations "${PERF_ITERATIONS}" \
    --warmup "${PERF_WARMUP}" \
    --rounds "${PERF_ROUNDS}" \
    --threshold-percent "${PERF_THRESHOLD_PERCENT}" \
    --output "${PERF_OUTPUT_DIR}/putmem_on_stream_${NPROC_PER_NODE}pe.json"
