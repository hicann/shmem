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
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "aclshmem.h"
#include "aclshmemi_host_common.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int message_length);
void copy_demo(uint32_t block_dim, void* stream, uint8_t* src, uint8_t* dst, int elements);
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;
int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    return ACLSHMEM_SUCCESS;
}
int test_aclshmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 初始化ACL和ACLSHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    const int num10 = 10;
    const uint32_t mem_size = 1024UL * 1024UL;
    const uint32_t half_mem_size = 512UL * 1024UL;

    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);

    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint8_t *ptr = static_cast<uint8_t*>(aclshmem_malloc(mem_size));
    uint8_t *ptr_A = ptr + half_mem_size;

    // 初始化数据
    uint32_t trans_size = 32UL * 1024UL;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + num10);
    }

    status = aclrtMemcpy(ptr + aclshmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
                         input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // AllGather
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size * sizeof(int32_t));

    copy_demo(1, stream, ptr, ptr_A, n_ranks * trans_size * sizeof(int32_t));

    status = aclrtSynchronizeStream(stream);

    // 校验NPU的内容
    if (rank_id <= n_ranks) {
        int32_t *y_host;
        size_t input_size = n_ranks * trans_size * sizeof(int32_t);

        // 校验 ptr_A 中的内容
        status = aclrtMallocHost(reinterpret_cast<void **>(&y_host), input_size);
        status = aclrtMemcpy(y_host, input_size, ptr_A, input_size, ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "Rank " << rank_id << " AllGather result in ptr_A without handle_wait:" << std::endl;
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
    aclshmem_free(ptr);
    status = aclshmem_finalize();
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
    status = test_aclshmem_team_all_gather(rank_id, n_ranks, local_mem_size);
    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}