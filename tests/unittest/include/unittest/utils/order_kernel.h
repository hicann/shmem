/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ORDER_KERNEL_H
#define ORDER_KERNEL_H

#include <stdint.h>

constexpr uint64_t QUIET_COMPLETION_WORDS = 64;
constexpr uint64_t QUIET_COMPLETION_REPEATS = 3;
constexpr uint64_t QUIET_COMPLETION_ROUND_SHIFT = 48U;
constexpr uint64_t QUIET_COMPLETION_PEER_SHIFT = 32U;

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
__aicore__
#endif
    inline constexpr uint64_t
    quiet_completion_value(uint64_t round, uint64_t peer, uint64_t index)
{
    return (round << QUIET_COMPLETION_ROUND_SHIFT) | (peer << QUIET_COMPLETION_PEER_SHIFT) | index;
}

void quiet_order_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks);
void fence_order_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks);
void quiet_completion_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks);
void quiet_completion_mte_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks);
void quiet_completion_rdma_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks);

#endif // ORDER_KERNEL_H
