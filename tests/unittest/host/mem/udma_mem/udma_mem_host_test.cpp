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
#include <string>

#include "acl/acl.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern int test_global_ranks;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern int32_t test_udma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream* st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void test_udma_put(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_get(uint32_t block_dim, void* stream, uint8_t* gva);
extern void test_udma_put_signal(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr);
extern void test_udma_put_signal_sq_wrap(uint32_t block_dim, void* stream, uint8_t* workspace);

constexpr size_t UDMA_TEST_WRAP_WORKSPACE_SIZE = 5UL * 1024UL * 1024UL;
constexpr uint64_t UDMA_TEST_WRAP_GUARD_VALUE = 0xC0FFEE1234567890ULL;
constexpr size_t UDMA_TEST_WRAP_RESULT_COUNT = 8;
constexpr size_t UDMA_TEST_WRAP_PIPE_S_LAYOUT_IDX = 0;
constexpr size_t UDMA_TEST_WRAP_PIPE_MTE3_LAYOUT_IDX = 1;
constexpr size_t UDMA_TEST_WRAP_PIPE_S_GUARD_IDX = 2;
constexpr size_t UDMA_TEST_WRAP_PIPE_MTE3_GUARD_IDX = 3;
constexpr size_t UDMA_TEST_WRAP_FAA_LAYOUT_IDX = 4;
constexpr size_t UDMA_TEST_WRAP_CAS_LAYOUT_IDX = 5;
constexpr size_t UDMA_TEST_WRAP_FAA_GUARD_IDX = 6;
constexpr size_t UDMA_TEST_WRAP_CAS_GUARD_IDX = 7;

static void test_udma_put_get(aclrtStream stream, uint8_t* gva, uint32_t rank_id, uint32_t rank_size)
{
    size_t messageSize = 64;
    uint32_t rankOffset = 10;
    uint32_t* inHost;
    uint32_t* outHost;
    size_t totalSize = messageSize * rank_size;
    uint32_t block_dim = 1;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&inHost), totalSize), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&outHost), totalSize), 0);
    bzero(inHost, totalSize);
    for (uint32_t i = 0; i < messageSize / sizeof(uint32_t); i++) {
        inHost[i + rank_id * messageSize / sizeof(uint32_t)] = rank_id + rankOffset;
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_put(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_get(block_dim, stream, (uint8_t*)gva);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();
    ASSERT_EQ(aclrtMemcpy(outHost, totalSize, gva, totalSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < rank_size; i++) {
        ASSERT_EQ(outHost[i * messageSize / sizeof(uint32_t)], i + rankOffset);
    }

    // Test udma put signal
    uint8_t* sig_addr = static_cast<uint8_t*>(aclshmem_malloc(rank_size * sizeof(uint64_t)));
    ASSERT_NE(sig_addr, nullptr);

    ASSERT_EQ(aclrtMemcpy(gva, totalSize, inHost, totalSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_udma_put_signal(block_dim, stream, (uint8_t*)gva, sig_addr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    aclshmemi_control_barrier_all();

    // Read and validate signals
    std::vector<uint64_t> signal_values(rank_size, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            signal_values.data(), rank_size * sizeof(uint64_t), sig_addr, rank_size * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        0);

    std::cout << "Signal values received on rank=" << rank_id << ":" << std::endl;
    bool all_signals_set = true;
    for (int i = 0; i < rank_size; i++) {
        if (i == rank_id) {
            continue;
        }
        uint64_t signal = 1000;
        if (signal_values[i] != signal) {
            all_signals_set = false;
        }
    }
    ASSERT_TRUE(all_signals_set) << "Signal for rank " << rank_id << " is not set correctly";

    aclshmem_free(sig_addr);
    ASSERT_EQ(aclrtFreeHost(inHost), 0);
    ASSERT_EQ(aclrtFreeHost(outHost), 0);
}

void test_aclshmem_udma_mem(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    auto status = test_udma_init(rank_id, n_ranks, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    void* ptr = aclshmem_malloc(1024);
    test_udma_put_get(stream, (uint8_t*)ptr, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAMem)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    // test_mutil_task(test_aclshmem_udma_mem, local_mem_size, processCount);
}

void test_aclshmem_udma_put_signal_sq_wrap(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    const int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream = nullptr;
    ASSERT_EQ(test_udma_init(rank_id, n_ranks, local_mem_size, &stream), 0);
    ASSERT_NE(stream, nullptr);

    auto* workspace = static_cast<uint8_t*>(aclshmem_malloc(UDMA_TEST_WRAP_WORKSPACE_SIZE));
    ASSERT_NE(workspace, nullptr);
    ASSERT_EQ(aclrtMemset(workspace, UDMA_TEST_WRAP_WORKSPACE_SIZE, 0, UDMA_TEST_WRAP_WORKSPACE_SIZE), 0);

    test_udma_put_signal_sq_wrap(1, stream, workspace);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    std::vector<uint64_t> result(UDMA_TEST_WRAP_RESULT_COUNT, 0);
    ASSERT_EQ(
        aclrtMemcpy(
            result.data(), result.size() * sizeof(uint64_t), workspace, result.size() * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST),
        0);
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_S_LAYOUT_IDX], 1U) << "PIPE_S wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_MTE3_LAYOUT_IDX], 1U) << "PIPE_MTE3 wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_S_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE) << "PIPE_S wrote past the SQ ring";
    EXPECT_EQ(result[UDMA_TEST_WRAP_PIPE_MTE3_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE)
        << "PIPE_MTE3 wrote past the SQ ring";
    EXPECT_EQ(result[UDMA_TEST_WRAP_FAA_LAYOUT_IDX], 1U) << "FAA wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_CAS_LAYOUT_IDX], 1U) << "CAS wrap WQE layout is invalid";
    EXPECT_EQ(result[UDMA_TEST_WRAP_FAA_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE) << "FAA wrote past the SQ ring";
    EXPECT_EQ(result[UDMA_TEST_WRAP_CAS_GUARD_IDX], UDMA_TEST_WRAP_GUARD_VALUE) << "CAS wrote past the SQ ring";

    aclshmem_free(workspace);
    test_finalize(stream, device_id);
}

TEST(TestMemApi, TestShmemUDMAPutSignalSqWrap)
{
    constexpr int min_rank_size = 2;
    if (test_global_ranks < min_rank_size) {
        GTEST_SKIP() << "UDMA SQ wrap test requires at least " << min_rank_size << " ranks";
    }
    if (test_gnpu_num < test_global_ranks) {
        GTEST_SKIP() << "UDMA SQ wrap test requires at least " << test_global_ranks << " NPUs";
    }
    const char* soc_name = aclrtGetSocName();
    if (soc_name == nullptr || std::string(soc_name).find("Ascend950") == std::string::npos) {
        GTEST_SKIP() << "UDMA SQ wrap test requires Ascend950";
    }
    const uint64_t local_mem_size = 1024UL * 1024UL * 1024UL;
    test_mutil_task(test_aclshmem_udma_put_signal_sq_wrap, local_mem_size, test_global_ranks);
}
