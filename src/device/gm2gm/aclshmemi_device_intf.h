/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_DEVICE_INTF_H
#define ACLSHMEMI_DEVICE_INTF_H

#include "stdint.h"
#include "host_device/aclshmem_common_types.h"

// internal kernels
int32_t aclshmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count);

int32_t aclshmemi_barrier_on_stream(aclshmem_team_t tid, void *stream);

void aclshmemi_handle_wait_on_stream(aclshmem_handle_t handle, aclrtStream stream);

#endif