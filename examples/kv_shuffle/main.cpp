/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "fp16_t.h"
#include "bfloat16.h"
#include "utils.h"
#include "param.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "aclshmem.h"
#include "kv_shuffle_kernel.h"
int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *data_type;

constexpr int64_t MAX_SEQLEN = 1024;
constexpr int64_t MAX_BATCH = 10;
constexpr int64_t page_size = 128;
constexpr int64_t max_block_nums = MAX_SEQLEN * MAX_BATCH / page_size;
constexpr int64_t kv_head_num = 8;
constexpr int64_t head_dim = 128;

int test_shmem_kv_shuffle(int rank_id, int n_ranks)
{
    // ACLStream init
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    uint32_t BLOCK_NUM = 16;

    int64_t kv_cache_size = max_block_nums * kv_head_num * page_size * head_dim * sizeof(int8_t);

    std::string inputFile;
    // k_cache input
    uint8_t *k_cache_host;
    aclrtMallocHost(reinterpret_cast<void **>(&k_cache_host), kv_cache_size);
    inputFile = "../../examples/kv_shuffle/scripts/output/k_cache_input_rank_" + std::to_string(rank_id) + ".bin";
    ReadFile(inputFile, k_cache_host, kv_cache_size);
    void *k_cache_ptr = shmem_malloc(kv_cache_size);
    aclrtMemcpy(k_cache_ptr, kv_cache_size, k_cache_host, kv_cache_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // v_cache input
    uint8_t *v_cache_host;
    aclrtMallocHost(reinterpret_cast<void **>(&v_cache_host), kv_cache_size);
    inputFile = "../../examples/kv_shuffle/scripts/output/v_cache_input_rank_" + std::to_string(rank_id) + ".bin";
    ReadFile(inputFile, v_cache_host, kv_cache_size);
    void *v_cache_ptr = shmem_malloc(kv_cache_size);
    aclrtMemcpy(v_cache_ptr, kv_cache_size, v_cache_host, kv_cache_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // global_shuffle_table input
    uint8_t *global_shuffle_table_host;
    constexpr uint32_t PAIR_PER_RANK = 2;
    aclrtMallocHost(reinterpret_cast<void **>(&global_shuffle_table_host), n_ranks * PAIR_PER_RANK * sizeof(int64_t));
    inputFile = "../../examples/kv_shuffle/scripts/output/pair_list.bin";
    ReadFile(inputFile, global_shuffle_table_host, n_ranks * PAIR_PER_RANK * sizeof(int64_t));
    void *global_shuffle_table_ptr;
    aclrtMalloc(&global_shuffle_table_ptr, n_ranks * PAIR_PER_RANK * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(global_shuffle_table_ptr, n_ranks * PAIR_PER_RANK * sizeof(int64_t),
                global_shuffle_table_host, n_ranks * PAIR_PER_RANK * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // global_block_num input
    uint8_t *global_block_num_host;
    aclrtMallocHost(reinterpret_cast<void **>(&global_block_num_host), sizeof(int64_t));
    inputFile = "../../examples/kv_shuffle/scripts/output/block_num_rank_" + std::to_string(rank_id) + ".bin";
    ReadFile(inputFile, global_block_num_host, sizeof(int64_t));
    void *global_block_num_ptr;
    aclrtMalloc(&global_block_num_ptr, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(global_block_num_ptr, sizeof(int64_t), global_block_num_host,
                sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);

    const int64_t block_nums = *reinterpret_cast<int64_t *>(global_block_num_host);

    // src_block_table input
    uint8_t *src_block_table_host;
    void *src_block_table_ptr;
    if (block_nums != 0) {
        aclrtMallocHost(reinterpret_cast<void **>(&src_block_table_host), block_nums * sizeof(int64_t));
        inputFile = "../../examples/kv_shuffle/scripts/output/src_block_table_rank_" + std::to_string(rank_id) + ".bin";

        aclrtMalloc(&src_block_table_ptr, block_nums * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputFile, src_block_table_host, block_nums * sizeof(int64_t));
        aclrtMemcpy(src_block_table_ptr, block_nums * sizeof(int64_t),
                    src_block_table_host, block_nums * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::cout << "Rank " << rank_id << " block_nums = 0, Skip src_block_table input" << std::endl;
    }

    // dst_block_table input
    uint8_t *dst_block_table_host;
    void *dst_block_table_ptr;
    if (block_nums != 0) {
        aclrtMallocHost(reinterpret_cast<void **>(&dst_block_table_host), block_nums * sizeof(int64_t));
        inputFile = "../../examples/kv_shuffle/scripts/output/dst_block_table_rank_" + std::to_string(rank_id) + ".bin";

        aclrtMalloc(&dst_block_table_ptr, block_nums * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(inputFile, dst_block_table_host, block_nums * sizeof(int64_t));
        aclrtMemcpy(dst_block_table_ptr, block_nums * sizeof(int64_t),
                    dst_block_table_host, block_nums * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        std::cout << "Rank " << rank_id << " block_nums = 0, Skip dst_block_table input" << std::endl;
    }

    // KVShuffle
    KVShuffleOps ops(BLOCK_NUM, stream);
    int PERF_TIMES = 10;
    for (int zz = 0; zz < PERF_TIMES; zz++) {
        ops.compute(
            (uint8_t *)k_cache_ptr,
            (uint8_t *)v_cache_ptr,
            (uint8_t *)global_shuffle_table_ptr,
            (uint8_t *)src_block_table_ptr,
            (uint8_t *)dst_block_table_ptr,
            block_nums,
            kv_head_num, page_size, head_dim);
    }
    status = aclrtSynchronizeStream(stream);

    // Result Check
    std::string outputFile;
    int8_t *k_output_host;
    status = aclrtMallocHost(reinterpret_cast<void**>(&k_output_host), kv_cache_size);
    status = aclrtMemcpy(k_output_host, kv_cache_size, k_cache_ptr, kv_cache_size, ACL_MEMCPY_DEVICE_TO_HOST);
    outputFile = "../../examples/kv_shuffle/scripts/output/k_cache_output_rank_" + std::to_string(rank_id) + ".bin";
    WriteFile(outputFile, k_output_host, kv_cache_size);

    int8_t *v_output_host;
    status = aclrtMallocHost(reinterpret_cast<void**>(&v_output_host), kv_cache_size);
    status = aclrtMemcpy(v_output_host, kv_cache_size, v_cache_ptr, kv_cache_size, ACL_MEMCPY_DEVICE_TO_HOST);
    outputFile = "../../examples/kv_shuffle/scripts/output/v_cache_output_rank_" + std::to_string(rank_id) + ".bin";
    WriteFile(outputFile, v_output_host, kv_cache_size);

    shmem_free(k_cache_ptr);
    shmem_free(v_cache_ptr);
    aclrtFree(global_shuffle_table_ptr);
    aclrtFree(global_block_num_ptr);
    if (block_nums > 0) {
        aclrtFree(src_block_table_ptr);
        aclrtFree(dst_block_table_ptr);
    }

    status = aclrtFreeHost(k_cache_host);
    status = aclrtFreeHost(v_cache_host);
    status = aclrtFreeHost(global_shuffle_table_host);
    status = aclrtFreeHost(global_block_num_host);
    if (block_nums > 0) {
        status = aclrtFreeHost(src_block_table_host);
        status = aclrtFreeHost(dst_block_table_host);
    }
    status = aclrtFreeHost(k_output_host);
    status = aclrtFreeHost(v_output_host);

    status = aclrtDestroyStream(stream);
    return status;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_ranks = atoi(argv[INDEX1]);
    int rank_id = atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    // int32_t ret = shmem_set_conf_store_tls(false, nullptr, 0);

    // Acl && Shmem init
    int32_t device_id = rank_id % g_npus + f_npu;
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    shmem_init_attr_t *attributes;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);

    status = test_shmem_kv_shuffle(rank_id, n_ranks);

    status = shmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    return 0;
}
