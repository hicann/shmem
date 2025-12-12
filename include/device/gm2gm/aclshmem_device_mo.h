/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_DEVICE_MO_H
#define SHMEM_DEVICE_MO_H

#include "host_device/common_types.h"
#include "internal/device/sync/shmemi_device_quiet.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The shmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
 *        calling PE.
 *        On systems with only scale-up network (HCCS), updates are globally visible, whereas on systems with
 *        both scale-up network HCCS and scale-out network (RDMA), SHMEM only guarantees that updates to the
 *        memory of a given PE are visible to that PE.
 *        Quiet operations issued on the CPU and the NPU only complete communication operations that were
 *        issued from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from
 *        the CPU, using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
SHMEM_DEVICE void shmem_quiet()
{
    shmemi_quiet();
}

/**
 * @brief In OpenSHMEM specification, shmem_fence assures ordering of delivery of Put, AMOs, and memory store routines
 *        to symmetric data objects, but does not guarantee the completion of these operations.
 *        However, due to hardware capabilities, we implemented shmem_fence same as shmem_quiet, ensuring both ordering
 *        and completion.
 *        Fence operations issued on the CPU and the NPU only order communication operations that were issued from the
 *        CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU,
 *        using aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 */
SHMEM_DEVICE void shmem_fence()
{
    shmemi_quiet();
}

#ifdef __cplusplus
}
#endif

#endif