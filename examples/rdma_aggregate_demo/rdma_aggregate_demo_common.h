/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_AGGREGATE_DEMO_COMMON_H
#define RDMA_AGGREGATE_DEMO_COMMON_H

#include <cstdint>

namespace rdma_aggregate_demo {

constexpr uint32_t kElementCount = 16;
constexpr uint32_t kSlotCount = 26;
constexpr uint32_t kPutPointerLoopDeferOperationCount = 5;
constexpr uint32_t kPutPointerLoopDeferUbBytes = 64 + 128 * kPutPointerLoopDeferOperationCount;
constexpr uint32_t kAggregateMaxUbBytes = kPutPointerLoopDeferUbBytes;
constexpr uint32_t kSyncId = 0;

constexpr uint32_t kGetPointerSource0 = 0;
constexpr uint32_t kGetPointerDestination0 = 1;
constexpr uint32_t kGetPointerSource1 = 2;
constexpr uint32_t kGetPointerDestination1 = 3;
constexpr uint32_t kGetTensorSource0 = 4;
constexpr uint32_t kGetTensorDestination0 = 5;
constexpr uint32_t kGetTensorSource1 = 6;
constexpr uint32_t kGetTensorDestination1 = 7;
constexpr uint32_t kPutPointerSource0 = 8;
constexpr uint32_t kPutPointerDestination0 = 9;
constexpr uint32_t kPutPointerSource1 = 10;
constexpr uint32_t kPutPointerDestination1 = 11;
constexpr uint32_t kPutTensorSource0 = 12;
constexpr uint32_t kPutTensorDestination0 = 13;
constexpr uint32_t kPutTensorSource1 = 14;
constexpr uint32_t kPutTensorDestination1 = 15;
constexpr uint32_t kPutPointerLoopDeferSource0 = 16;
constexpr uint32_t kPutPointerLoopDeferDestination0 = 17;
constexpr uint32_t kPutPointerLoopDeferSource1 = 18;
constexpr uint32_t kPutPointerLoopDeferDestination1 = 19;
constexpr uint32_t kPutPointerLoopDeferSource2 = 20;
constexpr uint32_t kPutPointerLoopDeferDestination2 = 21;
constexpr uint32_t kPutPointerLoopDeferSource3 = 22;
constexpr uint32_t kPutPointerLoopDeferDestination3 = 23;
constexpr uint32_t kPutPointerLoopDeferSource4 = 24;
constexpr uint32_t kPutPointerLoopDeferDestination4 = 25;

constexpr uint32_t value_for(uint32_t pe, uint32_t slot, uint32_t element)
{
    return (pe + 1) * 100000U + slot * 1000U + element;
}

} // namespace rdma_aggregate_demo

#endif // RDMA_AGGREGATE_DEMO_COMMON_H
