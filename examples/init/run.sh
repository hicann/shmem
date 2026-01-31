#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

# 初始化默认值
MODE="default"  # 默认模式为default
NUM_PROCESSES=2 # 默认进程数为2

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -mode)
            MODE="$2"
            shift 2
            ;;
        -pesize)
            NUM_PROCESSES="$2"
            if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: pesize must be a positive integer!"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Error: Unknown parameter '$1'"
            echo "Usage: $0 [-mode <default|mpi|uid>] [-pesize <num>]"
            exit 1
            ;;
    esac
done

case "$MODE" in
    default)
        MODE_ID=1
        ;;
    mpi)
        MODE_ID=2
        ;;
    uid)
        MODE_ID=3
        ;;
    *)
        echo "Error: Invalid mode '$MODE'! Only 'default'/'mpi'/'uid' are allowed"
        echo "Usage: $0 [-mode <default|mpi|uid>] [-pesize <num>]"
        exit 1
        ;;
esac

BUILD_DIR="build"
EXECUTABLE_NAME="init_examples"

echo "=== Prepare build directory ==="
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"/*
fi
mkdir -p "$BUILD_DIR"

echo "=== Run CMake with RUN_MODE=${MODE_ID} (mode: ${MODE}) ==="
cd "$BUILD_DIR" || { echo "Error: Failed to enter build directory!"; exit 1; }
cmake -DRUN_MODE="${MODE_ID}" ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed!"
    exit 1
fi

echo "=== Compile executable (mode: ${MODE}, pesize: ${NUM_PROCESSES}) ==="
make -j$(nproc) "${EXECUTABLE_NAME}"
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi
cd ..

echo "=== Launch executable (mode: ${MODE}, pesize: ${NUM_PROCESSES}) ==="
export SHMEM_UID_SESSION_ID=127.0.0.1:8666
IPPORT="tcp://127.0.0.1:8666"

case "$MODE" in
    mpi|uid)
        mpirun -np "$NUM_PROCESSES" ./build/bin/"${EXECUTABLE_NAME}"
        ;;
    default)
        for (( idx=0; idx < NUM_PROCESSES; idx++ )); do
            echo "Starting process $idx/$NUM_PROCESSES..."
            ./build/bin/"${EXECUTABLE_NAME}" "$idx" "$NUM_PROCESSES" "${IPPORT}" &
        done
        wait
        echo "=== All processes completed ==="
        ;;
esac

if [ $? -eq 0 ]; then
    echo "=== Execution succeeded (mode: ${MODE}, pesize: ${NUM_PROCESSES})! ==="
else
    echo "=== Execution failed (mode: ${MODE}, pesize: ${NUM_PROCESSES})! ==="
    exit 1
fi