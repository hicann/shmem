/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "aclshmemi_host_common.h"
#include "gm2gm/aclshmemi_device_intf.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *config, uint32_t *len);

static uint64_t ffts_config;

int32_t aclshmemi_sync_init()
{
    uint32_t len;
    return rtGetC2cCtrlAddr(&ffts_config, &len);
}

uint64_t util_get_ffts_config()
{
    return ffts_config;
}

void aclshmem_barrier(aclshmem_team_t tid)
{
    // using default stream to do barrier
    aclshmemi_barrier_on_stream(tid, nullptr);
}

void aclshmem_barrier_all()
{
    aclshmem_barrier(ACLSHMEM_TEAM_WORLD);
}

void aclshmemx_barrier_on_stream(aclshmem_team_t tid, aclrtStream stream)
{
    aclshmemi_barrier_on_stream(tid, stream);
}

void aclshmemx_barrier_all_on_stream(aclrtStream stream)
{
    aclshmemi_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream);
}

void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream)
{
    aclshmemi_handle_wait_on_stream(handle, stream);
}
