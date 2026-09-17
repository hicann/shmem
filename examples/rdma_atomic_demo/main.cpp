/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"
#include "rdma_atomic_demo_kernel.h"

static int parse_int_arg(const char* text, const char* name)
{
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid " << name << ": " << text << std::endl;
        return -1;
    }
    return static_cast<int>(parsed);
}

int test_aclshmem_rdma_atomic(int pe_id, int n_pes, const char* ipport, int device_id)
{
    // Initialize ACL and ACLSHMEM.
    int status = 0;
    aclrtStream stream = nullptr;
    status = aclInit(nullptr);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclInit failed, status=%d", status), return status);
    status = aclrtSetDevice(device_id);
    if (status != 0) {
        ERROR_LOG("aclrtSetDevice failed, status=%d", status);
        (void)aclFinalize();
        return status;
    }
    status = aclrtCreateStream(&stream);
    if (status != 0) {
        ERROR_LOG("aclrtCreateStream failed, status=%d", status);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }

    aclshmemx_uniqueid_t default_flag_uid{};
    aclshmemx_init_attr_t attributes{};
    const uint64_t local_mem_size = 64UL * 1024UL * 1024UL;
    status = test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    if (status != 0) {
        ERROR_LOG("test_set_attr failed, status=%d", status);
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_ROCE;
    // Default bootstrap uses the command-line IP and port.
    attributes.comm_args = nullptr;
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (status != 0) {
        ERROR_LOG("aclshmemx_init_attr failed, status=%d", status);
        (void)aclrtDestroyStream(stream);
        (void)aclrtResetDevice(device_id);
        (void)aclFinalize();
        return status;
    }
    // A fatal error after SHMEM initialization may leave other PEs in a different collective.
    // Return without collective free/finalize in that case; cleanup below is for completed kernels.
    status = aclshmemx_set_rdma_config(0, UB_BYTES, 0);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclshmemx_set_rdma_config failed, status=%d", status), return status);

    auto buffer = static_cast<uint8_t*>(aclshmem_malloc(BUFFER_BYTES));
    if (buffer == nullptr) {
        std::cerr << "[ERROR] pe=" << pe_id << " aclshmem_malloc failed" << std::endl;
        return 1;
    }

    // Initialize the target and space for the returned values.
    uint64_t values[BUFFER_BYTES / sizeof(uint64_t)] = {};
    values[0] = INITIAL_VALUE + pe_id;
    status = aclrtMemcpy(buffer, BUFFER_BYTES, values, BUFFER_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("copy input failed, status=%d", status), return status);

    // Launch one AIV to perform FAA and CAS on the next PE.
    launch_rdma_atomic_demo(1, stream, buffer, buffer + RESULT_OFFSET);
    status = aclrtSynchronizeStream(stream);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("aclrtSynchronizeStream failed, status=%d", status), return status);

    // Copy back and validate the returned values and local target.
    status = aclrtMemcpy(values, BUFFER_BYTES, buffer, BUFFER_BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_CHECK_WITH_RET(status, ERROR_LOG("copy result failed, status=%d", status), return status);
    const int peer = (pe_id + 1) % n_pes;
    const uint64_t* old = values + RESULT_OFFSET / sizeof(uint64_t);
    const uint64_t peer_initial = INITIAL_VALUE + peer;
    const uint64_t expected_target = INITIAL_VALUE + pe_id + 2 * ADD_VALUE;
    if (old[0] != peer_initial || old[1] != peer_initial + ADD_VALUE || old[2] != peer_initial + 2 * ADD_VALUE ||
        values[0] != expected_target) {
        std::cerr << "[ERROR] pe=" << pe_id << " expected faa_old=" << peer_initial
                  << " cas_old=" << peer_initial + ADD_VALUE << " cas_miss_old=" << peer_initial + 2 * ADD_VALUE
                  << " target=" << expected_target << std::endl;
        status = 1;
    }

    // Release resources.
    aclshmem_free(buffer);
    status |= aclshmem_finalize();
    status |= aclrtDestroyStream(stream);
    status |= aclrtResetDevice(device_id);
    status |= aclFinalize();
    std::cout << "pe=" << pe_id << " peer=" << peer << " faa_old=" << old[0] << " cas_old=" << old[1]
              << " cas_miss_old=" << old[2] << " target=" << values[0] << std::endl;
    return status;
}

int main(int argc, char* argv[])
{
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <n_pes> <pe_id> <ipport> <g_npus> <f_pe> <f_npu>" << std::endl;
        return 1;
    }
    int argIdx = 1;
    int n_pes = parse_int_arg(argv[argIdx++], "n_pes");
    int pe_id = parse_int_arg(argv[argIdx++], "pe_id");
    const char* ipport = argv[argIdx++];
    int g_npus = parse_int_arg(argv[argIdx++], "g_npus");
    int f_pe = parse_int_arg(argv[argIdx++], "f_pe");
    int f_npu = parse_int_arg(argv[argIdx++], "f_npu");
    if (n_pes < 2 || pe_id < 0 || pe_id >= n_pes) {
        std::cerr << "Invalid PE arguments; use at least two PEs and a pe_id in [0, n_pes)." << std::endl;
        return 1;
    }
    const int64_t device_id = static_cast<int64_t>(pe_id) - f_pe + f_npu;
    if (g_npus <= 0 || f_pe < 0 || f_npu < 0 || f_pe > pe_id || pe_id - f_pe >= g_npus ||
        device_id > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid PE/device mapping; use one process per NPU." << std::endl;
        return 1;
    }
    const int status = test_aclshmem_rdma_atomic(pe_id, n_pes, ipport, static_cast<int>(device_id));
    if (status != 0) {
        std::cerr << "[FAILED] demo run failed in pe " << pe_id << ", status=" << status << std::endl;
    } else {
        std::cout << "[SUCCESS] demo run success in relative pe " << pe_id << std::endl;
    }
    return status;
}
