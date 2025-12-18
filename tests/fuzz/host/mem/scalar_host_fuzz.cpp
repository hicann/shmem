/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>
#include <secodefuzz/secodeFuzz.h>
#include <sys/types.h>

#include "acl/acl_rt.h"
#include "bfloat16.h"
#include "fp16_t.h"
#include "host/aclshmem_host_def.h"
#include "host/mem/aclshmem_host_heap.h"
#include "host/init/aclshmem_host_init.h"
#include "host/data_plane/aclshmem_host_p2p_sync.h"
#include "aclshmem_fuzz.h"
#include "utils/func_type.h"

static constexpr float test_offset = 3.5f;

class ShmemScalarFuzz : public testing::Test {
public:
    void SetUp()
    {
        DT_Enable_Leak_Check(0, 0);
        DT_Set_Running_Time_Second(ACLSHMEM_FUZZ_RUNNING_SECONDS);
    }

    void TearDown()
    {
    }
};

#define EXTERN_ONE_NUM_DO(NAME, TYPE)                                                                    \
    extern void put_##NAME##_one_num_do(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, \
                                        uint8_t *dev);                                                   \
    extern void get_##NAME##_one_num_do(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev)

ACLSHMEM_FUNC_TYPE_HOST(EXTERN_ONE_NUM_DO);

using device_func = std::function<void(uint32_t block_dim, void *stream, uint64_t config, uint8_t *gva, uint8_t *dev)>;

template <class Tp>
static void test_aclshmem_scalar(int rank_id, int n_ranks, uint64_t local_mem_size, device_func put_func,
                              device_func get_func)
{
    aclshmem_init_scope scope(rank_id, n_ranks, local_mem_size);
    EXPECT_EQ(aclshmemx_init_status(), ACLSHMEM_STATUS_IS_INITIALIZED);

    Tp *y_host;
    size_t data_size = sizeof(Tp);
    EXPECT_EQ(aclrtMallocHost((void **)(&y_host), data_size), ACL_SUCCESS);

    void *dev_ptr;
    EXPECT_EQ(aclrtMalloc(&dev_ptr, data_size, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);

    Tp value = static_cast<Tp>(test_offset) + static_cast<Tp>(rank_id);
    EXPECT_EQ(aclrtMemcpy(dev_ptr, data_size, &value, data_size, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    void *ptr = aclshmem_malloc(data_size);
    EXPECT_NE(ptr, nullptr);

    uint32_t block_dim = 1;
    put_func(block_dim, scope.stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);
    EXPECT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(y_host, data_size, ptr, data_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(y_host[0], static_cast<Tp>(test_offset) + static_cast<Tp>((rank_id + n_ranks - 1) % n_ranks));

    get_func(block_dim, scope.stream, util_get_ffts_config(), (uint8_t *)ptr, (uint8_t *)dev_ptr);
    EXPECT_EQ(aclrtSynchronizeStream(scope.stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(y_host, data_size, dev_ptr, data_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(y_host[0], static_cast<Tp>(test_offset) + static_cast<Tp>(rank_id % n_ranks));

    aclshmem_free(ptr);
    EXPECT_EQ(aclrtFree(dev_ptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(y_host), ACL_SUCCESS);
}

#define TEST_CASE(NAME, TYPE)                                                                      \
    TEST_F(ShmemScalarFuzz, aclshmem_scalar_##NAME##_success)                                         \
    {                                                                                              \
        char fuzzName[] = "aclshmem_scalar_##NAME##_success";                                         \
        uint64_t seed = 0;                                                                         \
        DT_FUZZ_START(seed, ACLSHMEM_FUZZ_COUNT, fuzzName, 0)                                         \
        {                                                                                          \
            aclshmem_fuzz_multi_task(                                                                 \
                [](int rank_id, int n_ranks, uint64_t local_mem_size) {                            \
                    device_func put_func = put_##NAME##_one_num_do;                                \
                    device_func get_func = get_##NAME##_one_num_do;                                \
                    test_aclshmem_scalar<TYPE>(rank_id, n_ranks, local_mem_size, put_func, get_func); \
                },                                                                                 \
                1 * GiB, aclshmem_fuzz_gnpu_num());                                                   \
        }                                                                                          \
        DT_FUZZ_END()                                                                              \
    }

ACLSHMEM_FUNC_TYPE_HOST(TEST_CASE);
