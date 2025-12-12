/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <unordered_set>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "aclshmemi_host_common.h"
#include "unittest_main_test.h"

static uint8_t *const heap_memory_start = (uint8_t *)(ptrdiff_t)0x100000000UL;
static uint64_t heap_memory_size = 4UL * 1024UL * 1024UL;
static aclrtStream heap_memory_stream;

class ShareMemoryManagerTest : public testing::Test {
protected:
    void Initialize(int rank_id, int n_ranks, uint64_t local_mem_size)
    {
        uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
        int status = SHMEM_SUCCESS;
        EXPECT_EQ(aclInit(nullptr), 0);
        EXPECT_EQ(status = aclrtSetDevice(device_id), 0);

        shmem_init_attr_t *attributes;
        shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

        status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);
        EXPECT_EQ(status, SHMEM_SUCCESS);
        EXPECT_EQ(g_state.mype, rank_id);
        EXPECT_EQ(g_state.npes, n_ranks);
        EXPECT_NE(g_state.heap_base, nullptr);
        EXPECT_NE(g_state.host_p2p_heap_base[rank_id], nullptr);
        EXPECT_EQ(g_state.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
        EXPECT_NE(g_state.team_pools[0], nullptr);
        status = shmem_init_status();
        EXPECT_EQ(status, SHMEM_STATUS_IS_INITIALIZED);
        testingRank = true;
    }

    bool testingRank = false;
};

TEST_F(ShareMemoryManagerTest, allocate_zero)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(0UL);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, allocate_one_piece_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(4096UL);
            EXPECT_NE(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, allocate_full_space_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(heap_memory_size);
            EXPECT_NE(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, allocate_large_memory_failed)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(heap_memory_size + SHMEM_EXTRA_SIZE + 1UL);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, calloc_zero)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t nmemb = 16;
            auto ptr = static_cast<uint32_t *>(shmem_calloc(nmemb, 0UL));
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, calloc_one_piece_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t nmemb = 16;
            const size_t elemSize = sizeof(uint32_t);
            auto ptr = static_cast<uint32_t *>(shmem_calloc(nmemb, elemSize));
            EXPECT_NE(nullptr, ptr);
            uint32_t *ptr_host;
            ASSERT_EQ(aclrtMallocHost((void **)&ptr_host, sizeof(uint32_t) * nmemb), 0);
            ASSERT_EQ(aclrtMemcpy(ptr_host, sizeof(uint32_t) * nmemb, ptr, sizeof(uint32_t) * nmemb,
                                  ACL_MEMCPY_DEVICE_TO_HOST),
                      0);
            for (size_t i = 0; i < nmemb; ++i) {
                EXPECT_EQ(ptr_host[i], 0u);
            }
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, calloc_full_space_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t nmemb = 16;
            auto ptr = shmem_calloc(nmemb, heap_memory_size / nmemb);
            EXPECT_NE(nullptr, ptr);
            uint32_t *ptr_host;
            ASSERT_EQ(aclrtMallocHost((void **)&ptr_host, sizeof(uint32_t) * nmemb), 0);
            ASSERT_EQ(aclrtMemcpy(ptr_host, heap_memory_size, ptr, heap_memory_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
            for (size_t i = 0; i < nmemb; ++i) {
                EXPECT_EQ(ptr_host[i], 0u);
            }
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, calloc_large_memory_failed)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t nmemb = 16;
            auto ptr = shmem_calloc(nmemb, (heap_memory_size + SHMEM_EXTRA_SIZE) / nmemb + 1UL);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, calloc_multiply_overflow_size_t_max)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;

    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            const size_t nmemb = static_cast<size_t>(~0ULL);
            const size_t each  = 2;

            void *p = shmem_calloc(nmemb, each);
            EXPECT_EQ(nullptr, p);

            void *ok = shmem_malloc(4096UL);
            EXPECT_NE(nullptr, ok);
            shmem_free(ok);

            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, align_zero)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t alignment = 16;
            auto ptr = shmem_align(alignment, 0UL);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, align_one_piece_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t alignment = 16;
            const size_t size = 128UL;
            auto ptr = shmem_align(alignment, size);
            EXPECT_NE(nullptr, ptr);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) & (alignment - 1), 0u);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, align_full_space_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t alignment = 16;
            auto ptr = shmem_align(alignment, heap_memory_size);
            EXPECT_NE(nullptr, ptr);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) & (alignment - 1), 0u);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, align_large_memory_failed)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t alignment = 16;
            auto ptr = shmem_align(alignment, heap_memory_size + SHMEM_EXTRA_SIZE + 1UL);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, align_not_two_power_failed)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            const size_t alignment = 17;
            const size_t size = 128UL;
            auto ptr = shmem_align(alignment, size);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, stress_malloc_calloc_align_no_leak)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;

    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            constexpr int rounds = 100;
            std::vector<void*> ptrs;
            ptrs.reserve(rounds * 3);

            for (int i = 0; i < rounds; ++i) {
                void *p1 = shmem_malloc(1024UL + (i % 7) * 128UL);
                EXPECT_NE(nullptr, p1);
                ptrs.push_back(p1);

                void *p2 = shmem_calloc(32, 16 + (i % 5));
                EXPECT_NE(nullptr, p2);
                ptrs.push_back(p2);

                void *p3 = shmem_align(64, 1536UL + (i % 3) * 64UL);
                EXPECT_NE(nullptr, p3);
                ptrs.push_back(p3);

                if ((i % 4) == 0) {
                    shmem_free(p1);
                    ptrs[ptrs.size()-3] = nullptr;
                }
                if ((i % 6) == 0) {
                    shmem_free(p2);
                    ptrs[ptrs.size()-2] = nullptr;
                }
            }

            for (void *p : ptrs) {
                if (p) shmem_free(p);
            }

            void *big = shmem_malloc(heap_memory_size / 2);
            EXPECT_NE(nullptr, big);
            shmem_free(big);

            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, calls_before_init_and_after_finalize)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;

    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;

            EXPECT_EQ(nullptr, shmem_malloc(1024UL));
            EXPECT_EQ(nullptr, shmem_calloc(4, 256));
            EXPECT_EQ(nullptr, shmem_align(64, 4096UL));
            shmem_free(nullptr);

            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            void *ok = shmem_malloc(2048UL);
            EXPECT_NE(nullptr, ok);
            shmem_free(ok);
            test_finalize(stream, device_id);
    
            EXPECT_EQ(nullptr, shmem_malloc(1024UL));
            EXPECT_EQ(nullptr, shmem_calloc(2, 512));
            EXPECT_EQ(nullptr, shmem_align(32, 1024UL));
            shmem_free(nullptr);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, free_nullptr_is_noop)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;

    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            shmem_free(nullptr);

            void *p = shmem_malloc(8192UL);
            EXPECT_NE(nullptr, p);
            shmem_free(p);

            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, double_free_should_not_corrupt_heap)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;

    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            size_t sz = 64UL * 1024UL;
            void *p = shmem_malloc(sz);
            ASSERT_NE(nullptr, p);

            shmem_free(p);
            shmem_free(p);

            void *q = shmem_malloc(sz);
            EXPECT_NE(nullptr, q);
            shmem_free(q);

            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, free_middle_pointer_should_not_work_and_not_corrupt)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;

    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);

            size_t sz = 128UL * 1024UL;
            uint8_t *base = static_cast<uint8_t*>(shmem_malloc(sz));
            ASSERT_NE(nullptr, base);

            void *middle = base + 64;

            shmem_free(middle);

            shmem_free(base);

            void *again = shmem_malloc(sz);
            EXPECT_NE(nullptr, again);
            shmem_free(again);

            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, free_merge)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto size = 1024UL * 1024UL;  // 1MB

            auto ptr1 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr1);

            auto ptr2 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr2);

            auto ptr3 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr3);

            auto ptr4 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr4);

            shmem_free(ptr2);
            shmem_free(ptr4);

            auto ptr5 = shmem_malloc(size * 2UL);
            ASSERT_EQ(nullptr, ptr5);

            shmem_free(ptr3);

            auto ptr6 = shmem_malloc(size * 3UL);
            ASSERT_NE(nullptr, ptr6);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}