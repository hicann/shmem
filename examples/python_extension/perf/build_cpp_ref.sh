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
OUTPUT_SO="${SCRIPT_DIR}/libputmem_on_stream_cpp_ref.so"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first." >&2
    exit 1
fi

# The first package import may print the one-time environment check.  The
# requested path is always the last output line.
SHMEM_INCLUDE=$(shmem-config --include | tail -n 1)
SHMEM_LIB=$(shmem-config --lib | tail -n 1)

if [[ ! -d "${SHMEM_INCLUDE}" || ! -d "${SHMEM_LIB}" ]]; then
    echo "Unable to locate installed cann-shmem headers or libraries." >&2
    exit 1
fi

if [[ -n "${CXX:-}" ]]; then
    CXX_BIN="${CXX}"
elif command -v g++ >/dev/null 2>&1; then
    CXX_BIN=$(command -v g++)
elif command -v c++ >/dev/null 2>&1; then
    CXX_BIN=$(command -v c++)
else
    CXX_BIN=$(command -v bisheng)
fi

"${CXX_BIN}" \
    -std=c++17 \
    -O3 \
    -fPIC \
    -shared \
    "${SCRIPT_DIR}/putmem_on_stream_cpp.cpp" \
    -I"${SHMEM_INCLUDE}" \
    -I"${ASCEND_HOME_PATH}/include" \
    -L"${SHMEM_LIB}" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -lshmem \
    -lascendcl \
    -Wl,-rpath,"${SHMEM_LIB}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${OUTPUT_SO}"

echo "Built C++ reference helper: ${OUTPUT_SO}"
