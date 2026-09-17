/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RDMA_ATOMIC_DEMO_KERNEL_H
#define RDMA_ATOMIC_DEMO_KERNEL_H

#include <cstdint>

constexpr uint32_t BUFFER_BYTES = 256;
// Keep locally written return values on a separate cache line from the remote atomic target.
constexpr uint32_t RESULT_OFFSET = 128;
constexpr uint32_t UB_BYTES = 128;
constexpr uint64_t INITIAL_VALUE = 10;
constexpr uint64_t ADD_VALUE = 10;

void launch_rdma_atomic_demo(uint32_t block_dim, void* stream, uint8_t* target, uint8_t* result);

#endif // RDMA_ATOMIC_DEMO_KERNEL_H
