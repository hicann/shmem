// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <acl/acl.h>

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "host/data_plane/shmem_host_rma.h"

extern "C" int shmem_perf_putmem_on_stream_cpp(
    uintptr_t dst, uintptr_t src, size_t bytes, int32_t pe, uintptr_t stream, int32_t iterations, double* elapsed_us)
{
    if (dst == 0 || src == 0 || bytes == 0 || pe < 0 || iterations <= 0 || elapsed_us == nullptr) {
        return -1;
    }

    auto acl_stream = reinterpret_cast<aclrtStream>(stream);
    const auto start = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < iterations; ++i) {
        aclshmemx_putmem_on_stream(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), bytes, pe, acl_stream);
    }

    const aclError ret = aclrtSynchronizeStream(acl_stream);
    const auto end = std::chrono::steady_clock::now();
    if (ret != ACL_SUCCESS) {
        return static_cast<int>(ret);
    }

    *elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
    return 0;
}
