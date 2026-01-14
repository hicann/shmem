/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "acl/acl.h"
#include "shmem.h"
#include "shmemi_host_common.h"
#include "unittest_main_test.h"

// ==================== 所有rank环形测试EQ比较 ====================
static void test_signal_eq_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    EXPECT_EQ(aclrtMemset(sig_addr, sizeof(int32_t), 0, sizeof(int32_t)), 0);
    
    int32_t expected_signal = rank_id;
    
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_EQ, expected_signal, stream);
    int32_t host_value = -1;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] current value is " << host_value << ", waiting for == " << expected_signal << std::endl;
    int next_rank = (rank_id + 1) % n_ranks;
    int32_t my_signal = next_rank;
    aclshmemx_signal_op_on_stream(sig_addr, my_signal, ACLSHMEM_SIGNAL_SET, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Set rank " << next_rank << " to " << my_signal << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_EQ(host_value, expected_signal);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " == " << expected_signal << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

static void test_signal_ne_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    int32_t init_value = (rank_id == 0) ? 100 : 0;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    int32_t wait_condition = 0;
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_NE, wait_condition, stream);
    int32_t host_value = -1;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);
    
    std::cout << "[RANK " << rank_id << "] current value is " << host_value << ", waiting for != " << wait_condition << std::endl;
    
    int next_rank = (rank_id + 1) % n_ranks;
    int32_t new_signal = 10 + rank_id;
    aclshmemx_signal_op_on_stream(sig_addr, new_signal, ACLSHMEM_SIGNAL_SET, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Set rank " << next_rank << " to " << new_signal << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_NE(host_value, wait_condition);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " != " << wait_condition << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

static void test_signal_gt_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    int32_t threshold = 10;
    int32_t init_value = (rank_id == 0) ? 20 : 0;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_GT, threshold, stream);
    int32_t host_value = -1;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] current value is " << host_value << ", waiting for > " << threshold << std::endl;
    
    int next_rank = (rank_id + 1) % n_ranks;
    int32_t large_value = threshold + rank_id + 1;
    aclshmemx_signal_op_on_stream(sig_addr, large_value, ACLSHMEM_SIGNAL_SET, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Set rank " << next_rank << " to " << large_value << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_GT(host_value, threshold);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " > " << threshold << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

static void test_signal_ge_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    int32_t base_value = 10;
    int32_t init_value = (rank_id == 0) ? 10 : 0;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_GE, base_value, stream);
    int32_t host_value = -1;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] current value is " << host_value << ", waiting for >= " << base_value << std::endl;
    
    int next_rank = (rank_id + 1) % n_ranks;
    int32_t larger_value = base_value + rank_id;
    aclshmemx_signal_op_on_stream(sig_addr, larger_value, ACLSHMEM_SIGNAL_SET, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Set rank " << next_rank << " to " << larger_value << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_GE(host_value, base_value);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " >= " << base_value << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

static void test_signal_lt_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    int32_t threshold = 500;
    int32_t init_value = (rank_id == 0) ? 400 : 600;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_LT, threshold, stream);
    int32_t host_value = -1;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] current value is " << host_value << ", waiting for < " << threshold << std::endl;
    
    int next_rank = (rank_id + 1) % n_ranks;
    int32_t small_value = threshold - rank_id - 1;
    aclshmemx_signal_op_on_stream(sig_addr, small_value, ACLSHMEM_SIGNAL_SET, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Set rank " << next_rank << " to " << small_value << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_LT(host_value, threshold);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " < " << threshold << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

static void test_signal_le_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    int32_t threshold = 500;
    int32_t init_value = (rank_id == 0) ? 500 : 600;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_LE, threshold, stream);
    int32_t host_value = -1;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] current value is " << host_value << ", waiting for <= " << threshold << std::endl;
    
    int next_rank = (rank_id + 1) % n_ranks;
    int32_t le_value = threshold - rank_id;
    aclshmemx_signal_op_on_stream(sig_addr, le_value, ACLSHMEM_SIGNAL_SET, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Set rank " << next_rank << " to " << le_value << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_LE(host_value, threshold);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " <= " << threshold << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

// ==================== 所有rank环形测试ADD操作 ====================
static void test_signal_add_all_ranks(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    aclrtStream stream;
    int64_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    EXPECT_NE(stream, nullptr);

    int32_t *sig_addr = static_cast<int32_t *>(aclshmem_malloc(sizeof(int32_t)));
    EXPECT_NE(sig_addr, nullptr);
    
    // 所有rank初始化为rank_id
    int init_value = rank_id;
    EXPECT_EQ(aclrtMemcpy(sig_addr, sizeof(int32_t), &init_value, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE), 0);
    
    int32_t host_value;
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] before atomicadd, host_value = " << host_value << std::endl;
    std::cout << "[TEST] signal ADD test rank " << rank_id << " of " << n_ranks << std::endl;

    int next_rank = (rank_id + 1) % n_ranks;
    
    // 每个rank都向后继rank的信号加一个唯一值
    int32_t add_value = 10;  
    aclshmemx_signal_op_on_stream(sig_addr, add_value, ACLSHMEM_SIGNAL_ADD, next_rank, stream);
    std::cout << "[RANK " << rank_id << "] Add " << add_value << " to rank " << next_rank << std::endl;
    
    sleep(1);
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << "[RANK " << rank_id << "] after add, host_value = " << host_value << std::endl;
    int32_t expected = rank_id + 10;
    
    // 每个rank都等待自己的信号达到expected
    aclshmemx_signal_wait_until_on_stream(sig_addr, ACLSHMEM_CMP_EQ, expected, stream);
    std::cout << "[RANK " << rank_id << "] Waiting for == " << expected << std::endl;
    
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    
    // 验证结果
    EXPECT_EQ(aclrtMemcpy(&host_value, sizeof(int32_t), sig_addr, sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    // 所有rank验证累加结果
    EXPECT_EQ(host_value, expected);
    std::cout << "[RANK " << rank_id << "] Verified: " << host_value << " == " << expected << std::endl;

    aclshmem_free(sig_addr);
    test_finalize(stream, device_id);
}

TEST(TEST_SIGNAL_API, test_signal_eq_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_eq_all_ranks, local_mem_size, process_count);
}

TEST(TEST_SIGNAL_API, test_signal_ne_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_ne_all_ranks, local_mem_size, process_count);
}

TEST(TEST_SIGNAL_API, test_signal_gt_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_gt_all_ranks, local_mem_size, process_count);
}

TEST(TEST_SIGNAL_API, test_signal_ge_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_ge_all_ranks, local_mem_size, process_count);
}

TEST(TEST_SIGNAL_API, test_signal_lt_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_lt_all_ranks, local_mem_size, process_count);
}

TEST(TEST_SIGNAL_API, test_signal_le_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_le_all_ranks, local_mem_size, process_count);
}

TEST(TEST_SIGNAL_API, test_signal_add_all_ranks)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_signal_add_all_ranks, local_mem_size, process_count);
}