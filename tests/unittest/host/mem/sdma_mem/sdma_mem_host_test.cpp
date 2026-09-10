/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "unittest/sdma_kernel.h"
#include "unittest_main_test.h"

// SDMA QP 接口 kernel 以 block_dim=1 启动（每 block 2 个 AIV），AIV 用 GetBlockIdx()
// （AIV 级全局索引 0~1）作为 qp_idx，须在初始化前按 block 数 × 每 block AIV 数创建 QP，
// 否则 AIV1 会触发设备侧 "SDMA QP index is out of range" 断言。
constexpr uint32_t SDMA_UT_BLOCK_DIM = 1;
constexpr uint32_t SDMA_UT_AIVS_PER_BLOCK = 2;
constexpr uint32_t SDMA_UT_QP_NUM = SDMA_UT_BLOCK_DIM * SDMA_UT_AIVS_PER_BLOCK;

static void test_sdma_put_get(aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size)
{
    size_t messageSize = 64;
    uint32_t rankOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * rank_size;
    constexpr uint32_t block_dim = SDMA_UT_BLOCK_DIM;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + rank_id * messageSize / sizeof(uint32_t)] = rank_id + rankOffset;
    }

    // Test SDMA Put (pointer interface)
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_sdma_put(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    // Test SDMA Get (pointer interface)
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_sdma_get(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    // Test SDMA Put with Tensor interface
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_sdma_put_tensor(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    // Test SDMA Get with Tensor interface
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_sdma_get_tensor(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    // Test SDMA Put without QP (single-core interface fixed on QP 0)
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_sdma_put_noqp(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    // Test SDMA Get without QP (single-core interface fixed on QP 0)
    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_sdma_get_noqp(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_sdma_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, SDMA_UT_QP_NUM), 0);
    aclrtStream stream;
    auto status = test_sdma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    size_t messageSize = 64;
    size_t totalSize = messageSize * n_ranks;
    void* ptr = aclshmem_malloc(totalSize);
    ASSERT_NE(ptr, nullptr);
    test_sdma_put_get(stream, (uint8_t*)ptr, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    aclshmem_free(ptr);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemSDMAMem)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_sdma_mem, local_mem_size, processCount);
}
