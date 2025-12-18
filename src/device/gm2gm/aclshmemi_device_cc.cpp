/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"

#include "aclshmem.h"
#include "host_device/aclshmem_common_types.h"
#include "aclshmemi_device_cc.h"
#include "aclshmemi_device_common.h"
#include "device/gm2gm/engine/aclshmem_device_rdma.h"

// kernels
ACLSHMEM_GLOBAL void k_aclshmem_barrier(int32_t tid)
{
    aclshmemi_barrier<false>(tid);
}

// interfaces
int32_t aclshmemi_barrier_on_stream(aclshmem_team_t tid, aclrtStream stream)
{
    // call barrier kernel
    k_aclshmem_barrier<<<1, nullptr, stream>>>((int32_t)tid);
    return aclrtSynchronizeStream(stream);
}

// kernels
ACLSHMEM_GLOBAL void k_aclshmem_handle_wait(int32_t tid)
{
    aclshmemi_handle(tid);
}

// interfaces
void aclshmemi_handle_wait_on_stream(aclshmem_handle_t handle, aclrtStream stream)
{
    // call barrier kernel
    k_aclshmem_handle_wait<<<1, nullptr, stream>>>((int32_t)handle.team_id);
}

// kernels
template <typename T>
ACLSHMEM_GLOBAL void k_memset(GM_ADDR array, int32_t len, T val, int32_t count)
{
    if (array == 0) {
        return;
    }
    auto tmp = (__gm__ T *)array;
    int32_t valid_count = count < len ? count : len;
    for (int32_t i = 0; i < valid_count; i++) {
        *tmp++ = val;
    }

    dcci_entire_cache();
}

// interfaces
int32_t aclshmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count)
{
    k_memset<int32_t><<<1, nullptr, nullptr>>>((uint8_t *)array, len, val, count);
    return aclrtSynchronizeStream(nullptr);
}