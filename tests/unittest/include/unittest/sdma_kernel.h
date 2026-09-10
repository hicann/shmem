/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SDMA_KERNEL_H
#define SDMA_KERNEL_H

#include <cstdint>

// sdma_mem UT kernels
void test_sdma_put(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_sdma_get(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_sdma_put_tensor(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_sdma_get_tensor(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_sdma_put_noqp(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_sdma_get_noqp(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);

// notifywait UT kernels
void test_put_notify_wait(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_get_notify_wait(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_put_tensor_notify_wait(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_get_tensor_notify_wait(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_put_notify_wait_noqp(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void test_get_notify_wait_noqp(uint32_t block_dim, void* stream, uint8_t* gva, uint64_t config);
void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements);

// cmo UT kernels
void copy_perf_kernel(
    uint32_t block_dim, void* stream, uint8_t* src, uint8_t* res, uint32_t copypad_size, uint32_t copypad_times);
void cmo_pretech_kernel(uint8_t* src, uint32_t size, void* stream);
void cmo_pretech_noqp_kernel(uint8_t* src, uint32_t size, void* stream);

#endif // SDMA_KERNEL_H
