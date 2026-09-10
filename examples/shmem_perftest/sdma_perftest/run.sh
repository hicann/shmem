#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should have received a copy of the License along with this program.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))
EXEC_BIN=${PROJECT_ROOT}/build/bin/sdma_perftest
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH:-}/lib64:${LD_LIBRARY_PATH:-}
cd "$SCRIPT_DIR"

TEST_TYPE=get
DATA_TYPE=float
MIN_EXPONENT=3
MAX_EXPONENT=17
LOOP_COUNT=1000
UB_SIZE=16
QP_NUM=2
METRIC=bw
BATCH=512
PE_SIZE=2
IPPORT=tcp://127.0.0.1:8769
GNPU_NUM=2
FIRST_NPU=0
FIRST_PE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-type) TEST_TYPE=$2; shift 2 ;;
        -d|--datatype) DATA_TYPE=$2; shift 2 ;;
        -e|--exponent) MIN_EXPONENT=$2; MAX_EXPONENT=$2; shift 2 ;;
        --exponent-range) MIN_EXPONENT=$2; MAX_EXPONENT=$3; shift 3 ;;
        --loop-count) LOOP_COUNT=$2; shift 2 ;;
        --ub-size) UB_SIZE=$2; shift 2 ;;
        -pes) PE_SIZE=$2; shift 2 ;;
        -ipport) IPPORT=$2; shift 2 ;;
        -gnpus) GNPU_NUM=$2; shift 2 ;;
        -fnpu) FIRST_NPU=$2; shift 2 ;;
        -fpe) FIRST_PE=$2; shift 2 ;;
        --qp) QP_NUM=$2; shift 2 ;;
        --metric) METRIC=$2; shift 2 ;;
        --batch) BATCH=$2; shift 2 ;;
        *) echo "Usage: $0 [-t get|bi_get|put|bi_put] [--qp qp_count] [-d datatype] [-e exponent|--exponent-range min max] [--loop-count N] [--metric bw|lat] [--batch N] [-pes N]"; exit 1 ;;
    esac
done

if [[ ! $QP_NUM =~ ^[0-9]+$ || $QP_NUM -lt 1 || $QP_NUM -gt 72 ]]; then echo "Error: --qp must be in [1, 72]"; exit 1; fi
if [[ ! $METRIC =~ ^(bw|lat)$ ]]; then echo "Error: --metric must be 'bw' or 'lat' (got '$METRIC')"; exit 1; fi
if [[ ! $BATCH =~ ^[0-9]+$ ]]; then echo "Error: --batch must be a non-negative integer (got '$BATCH')"; exit 1; fi

if [[ $TEST_TYPE != get && $TEST_TYPE != bi_get && $TEST_TYPE != put && $TEST_TYPE != bi_put ]]; then echo "Error: sdma_perftest supports -t get, bi_get, put or bi_put"; exit 1; fi
if [[ ! " float int8 int16 int32 int64 uint8 uint16 uint32 uint64 char " =~ " $DATA_TYPE " ]]; then echo "Error: unsupported datatype"; exit 1; fi
if [[ ! -x $EXEC_BIN ]]; then echo "Error: $EXEC_BIN does not exist; build with bash scripts/build.sh -examples"; exit 1; fi
export SHMEM_CYCLE_PROF_PE=${SHMEM_CYCLE_PROF_PE:-0}
mkdir -p "$SCRIPT_DIR/output"
PIDS=()
for ((pe = 0; pe < GNPU_NUM; ++pe)); do
    "$EXEC_BIN" --pes "$PE_SIZE" --pe-id "$pe" --ipport "$IPPORT" --gnpus "$GNPU_NUM" --fpe "$FIRST_PE" --fnpu "$FIRST_NPU" -t "$TEST_TYPE" --qp "$QP_NUM" -d "$DATA_TYPE" --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" --ub-size "$UB_SIZE" --metric "$METRIC" --batch "$BATCH" &
    PIDS+=("$!")
done
STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        STATUS=1
    fi
done
exit "$STATUS"
