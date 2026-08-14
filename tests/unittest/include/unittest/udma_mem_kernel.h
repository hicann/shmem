/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UDMA_MEM_KERNEL_H
#define UDMA_MEM_KERNEL_H

#include <cstdint>

void test_udma_put(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_get(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_put_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_put_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_get_action_pointer(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_get_action_tensor(uint32_t block_dim, void* stream, uint8_t* gva);
void test_udma_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_signal_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_size_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_size_signal_sync(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
void test_udma_highlevel_put_signal_split(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr, uint64_t elem_size, int32_t signal);

#endif // UDMA_MEM_KERNEL_H
