/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATOMIC_PERFTEST_COMMON_H
#define ATOMIC_PERFTEST_COMMON_H

#include <cstdint>
#include "perftest_common_types.h"

namespace perftest {

struct alignas(64) rdma_atomic_result_t {
    int64_t elapsed_cycles;
    uint64_t total_ops;
    uint64_t last_fetch_bits;
    uint64_t mismatch_count;
    uint64_t first_mismatch_index;
    uint64_t first_actual_bits;
    uint64_t expected_bits;
    uint64_t reserved;
};
static_assert(sizeof(rdma_atomic_result_t) == 64, "Each result occupies one cache line");

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define ATOMIC_PERF_INLINE __aicore__ inline
#else
#define ATOMIC_PERF_INLINE inline
#endif

ATOMIC_PERF_INLINE bool atomic_has_fetch(rdma_atomic_op_t op)
{
    return op == RDMA_ATOMIC_FETCH_INC || op == RDMA_ATOMIC_FETCH_ADD || op == RDMA_ATOMIC_FETCH_AND ||
           op == RDMA_ATOMIC_FETCH_OR || op == RDMA_ATOMIC_FETCH_XOR || op == RDMA_ATOMIC_SWAP ||
           op == RDMA_ATOMIC_COMPARE_SWAP;
}

// Convert through the unsigned type of the same width, so int32 never sign-extends into the result slot.
template <typename T>
ATOMIC_PERF_INLINE uint64_t atomic_value_bits(T value)
{
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        return static_cast<uint32_t>(value);
    } else {
        return static_cast<uint64_t>(value);
    }
}

template <typename T>
ATOMIC_PERF_INLINE T atomic_initial_value(rdma_atomic_op_t op)
{
    return op == RDMA_ATOMIC_AND || op == RDMA_ATOMIC_FETCH_AND ? T(-1) : T(0);
}

template <typename T>
ATOMIC_PERF_INLINE T atomic_operand(rdma_atomic_op_t op)
{
    return op == RDMA_ATOMIC_AND || op == RDMA_ATOMIC_FETCH_AND ? T(-2) : T(1);
}

template <typename T>
ATOMIC_PERF_INLINE uint64_t atomic_expected_bits(rdma_atomic_op_t op, int iterations)
{
    if (iterations == 0) {
        return atomic_value_bits(atomic_initial_value<T>(op));
    }
    switch (op) {
        case RDMA_ATOMIC_INC:
        case RDMA_ATOMIC_FETCH_INC:
        case RDMA_ATOMIC_ADD:
        case RDMA_ATOMIC_FETCH_ADD:
            return static_cast<uint64_t>(iterations);
        case RDMA_ATOMIC_AND:
        case RDMA_ATOMIC_FETCH_AND:
            return atomic_value_bits(T(-2));
        case RDMA_ATOMIC_XOR:
        case RDMA_ATOMIC_FETCH_XOR:
            return static_cast<uint64_t>(iterations % 2);
        default:
            return 1;
    }
}

template <typename T>
ATOMIC_PERF_INLINE uint64_t atomic_expected_fetch(rdma_atomic_op_t op, int iterations)
{
    return atomic_has_fetch(op) ? atomic_expected_bits<T>(op, iterations - 1) : 0;
}

#undef ATOMIC_PERF_INLINE
} // namespace perftest

extern "C" void launch_rdma_atomic_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t ffts_addr, uint8_t* target_gva, uint8_t* result_gva, uint64_t batch_size,
    int atomic_op, int data_type, int iteration_count, int sync_id);

#endif
