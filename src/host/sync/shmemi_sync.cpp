/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "gm2gm/shmemi_device_cc_kernel.h"

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

void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream)
{
    aclshmemi_handle_wait_on_stream(handle, stream);
}

void aclshmemx_signal_wait_until_on_stream(int32_t *sig_addr, int cmp, int32_t cmp_value, aclrtStream stream)
{
    call_aclshmemi_signal_wait_until_on_stream_kernel(sig_addr, cmp, cmp_value, stream);                            
}
