/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclshmem.h"
#include "aclshmemi_host_common.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length);
void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements);

int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 初始化ACL和SHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    const uint32_t mem_size = 1024UL * 1024UL;
    const uint32_t half_mem_size = 512UL * 1024UL;

    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    attributes->option_attr.data_op_engine_type = SHMEM_DATA_OP_ROCE;
    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);

    uint8_t *ptr = static_cast<uint8_t*>(shmem_malloc(mem_size));
    uint8_t *ptr_A = ptr + half_mem_size;

    // 初始化数据
    uint32_t trans_size = 32UL * 1024UL;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + num10);
    }

    status = aclrtMemcpy(ptr + shmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
                         input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // AllGather
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size * sizeof(int32_t));

    shmem_handle_t handle;
    handle.team_id = SHMEM_TEAM_WORLD;
    shmem_handle_wait(handle, stream);

    copy_demo(1, stream, ptr, ptr_A, n_ranks * trans_size * sizeof(int32_t));

    status = aclrtSynchronizeStream(stream);

    // 校验NPU的内容
    if (rank_id <= n_ranks) {
        int32_t *y_host;
        size_t input_size = n_ranks * trans_size * sizeof(int32_t);

        // 校验 ptr_A 中的内容
        status = aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);
        status = aclrtMemcpy(y_host, input_size, ptr_A, input_size, ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "Rank " << rank_id << " AllGather result in ptr_A after handle_wait:" << std::endl;
        int unexpected_count = 0;
        for (int i = 0; i < n_ranks; i++) {
            for (int j = 0; j < trans_size; j++) {
                if (y_host[trans_size * i + j] != num10 + i) {
                    unexpected_count++;
                }
            }
        }
        std::cout << "Rank " << rank_id << " has " << unexpected_count << " unexpected values." << std::endl;
        status = aclrtFreeHost(y_host);
    }

    // 去初始化
    shmem_free(ptr);
    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int argIdx = 1;
    int status = 0;
    int n_ranks = atoi(argv[argIdx++]);
    int rank_id = atoi(argv[argIdx++]);
    ipport = argv[argIdx++];
    g_npus = atoi(argv[argIdx++]);
    f_rank = atoi(argv[argIdx++]);
    f_npu = atoi(argv[argIdx++]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size);
    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}