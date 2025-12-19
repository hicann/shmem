/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_INIT_H
#define ACLSHMEMI_INIT_H

#include "stdint.h"
#include "host_device/aclshmem_common_types.h"
#include "init/init_backends/default/aclshmemi_init_default.h"
#include "init/init_backends/hybm/aclshmemi_init_hybm.h"

extern aclshmem_device_host_state_t g_state;
extern aclshmem_host_state_t g_state_host;

int32_t aclshmemi_control_barrier_all();

int32_t update_device_state(void);

int32_t aclshmemi_get_uniqueid_static_magic(aclshmemx_uniqueid_t *uid, bool is_root);

#endif  // ACLSHMEMI_INIT_H
