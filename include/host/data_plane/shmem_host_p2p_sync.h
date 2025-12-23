/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
    WARNING：

    Our barrier implementation ensures that:
        On systems with only HCCS: All operations of all pes of a team ON EXECUTING/INTERNAL STREAMs
        before the barrier are visiable to all pes of the team after the barrier.

    Refer to aclshmem_device_sync.h for using restrictions.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_HOST_API uint64_t util_get_ffts_config()
 * @brief Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel
 * using aclshmemx_set_ffts. Refer to aclshmemx_set_ffts for more details.
 *
 * @return ffts config
 */
ACLSHMEM_HOST_API uint64_t util_get_ffts_config();
#define shmemx_get_ffts_config util_get_ffts_config

/**
 * @fn ACLSHMEM_HOST_API void aclshmemx_handle_wait(aclshmem_handle_t handle)
 * @brief Wait asynchronous RMA operations to finish.
 */
ACLSHMEM_HOST_API void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream);
#define shmem_handle_wait aclshmemx_handle_wait

#ifdef __cplusplus
}
#endif

#endif