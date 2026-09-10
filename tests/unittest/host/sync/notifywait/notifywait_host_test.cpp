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

static void test_put_get_notify_wait(
    aclrtStream stream, uint8_t* gva, uint8_t* copy_ptr, uint32_t my_pe, uint32_t pe_size)
{
    size_t message_size = 64;
    uint32_t pe_offset = 10;
    uint32_t* in_host;
    uint32_t* out_host;
    size_t total_size = message_size * pe_size;
    constexpr uint32_t block_dim = SDMA_UT_BLOCK_DIM;

    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&in_host), total_size), 0);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&out_host), total_size), 0);
    bzero(in_host, total_size);
    for (uint32_t i = 0; i < message_size / sizeof(uint32_t); i++) {
        in_host[i + my_pe * message_size / sizeof(uint32_t)] = my_pe + pe_offset;
    }

    // Test SDMA Put (pointer interface) with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_put_notify_wait(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * SDMA_UT_AIVS_PER_BLOCK; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    aclshmem_barrier_all();
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t*)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Get (pointer interface) with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_get_notify_wait(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * SDMA_UT_AIVS_PER_BLOCK; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t*)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Put with Tensor interface with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_put_tensor_notify_wait(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * SDMA_UT_AIVS_PER_BLOCK; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    aclshmem_barrier_all();
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t*)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Get with Tensor interface with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_get_tensor_notify_wait(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    for (uint32_t i = 0; i < block_dim * SDMA_UT_AIVS_PER_BLOCK; i++) {
        aclrtWaitAndResetNotify(g_state_host.notify_arr[i], g_state_host.default_stream, 0);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t*)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Put without QP (single-core interface fixed on QP 0) with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_put_notify_wait_noqp(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    // 不带QP接口固定使用QP 0，仅产生一个notify，只需等待notify_arr[0]
    aclrtWaitAndResetNotify(g_state_host.notify_arr[0], g_state_host.default_stream, 0);
    aclshmem_barrier_all();
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t*)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    // Test SDMA Get without QP (single-core interface fixed on QP 0) with notify wait
    ASSERT_EQ(aclrtMemcpy(gva, total_size, in_host, total_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);
    aclshmemi_control_barrier_all();
    test_get_notify_wait_noqp(block_dim, stream, (uint8_t*)gva, util_get_ffts_config());
    aclrtWaitAndResetNotify(g_state_host.notify_arr[0], g_state_host.default_stream, 0);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    copy_demo(1, g_state_host.default_stream, (uint8_t*)gva, copy_ptr, total_size);
    ASSERT_EQ(aclrtSynchronizeStream(g_state_host.default_stream), 0);

    ASSERT_EQ(aclrtMemcpy(out_host, total_size, copy_ptr, total_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (uint32_t i = 0; i < pe_size; i++) {
        ASSERT_EQ(out_host[i * message_size / sizeof(uint32_t)], i + pe_offset);
    }

    ASSERT_EQ(aclrtFreeHost(in_host), 0);
    ASSERT_EQ(aclrtFreeHost(out_host), 0);
}

void test_aclshmem_notify_wait(int my_pe, int n_pes, uint64_t local_mem_size)
{
    int32_t device_id = my_pe % test_gnpu_num + test_first_npu;
    ASSERT_EQ(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, SDMA_UT_QP_NUM), 0);
    aclrtStream stream;
    auto status = test_sdma_init(my_pe, n_pes, local_mem_size, &stream);
    if (status != 0) {
        return;
    }
    ASSERT_NE(stream, nullptr);

    size_t message_size = 64;
    size_t total_size = message_size * n_pes;
    void* ptr = aclshmem_malloc(total_size * 2);
    ASSERT_NE(ptr, nullptr);
    // Set copy_ptr to point to the memory right after ptr (total_size bytes from ptr)
    uint8_t* copy_ptr = reinterpret_cast<uint8_t*>(ptr) + total_size;

    test_put_get_notify_wait(stream, (uint8_t*)ptr, copy_ptr, my_pe, n_pes);
    std::cout << "[TEST] begin to exit...... my_pe: " << my_pe << std::endl;

    // Free only ptr, as copy_ptr points to a part of ptr's memory
    aclshmem_free(ptr);
    test_finalize(stream, device_id);
}

TEST(TEST_SYNC_API, TestShmemNotifyWait)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 64;
    test_mutil_task(test_aclshmem_notify_wait, local_mem_size, process_count);
}
