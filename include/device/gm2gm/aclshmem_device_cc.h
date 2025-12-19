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
    WARNING：Restrictions of Barrier APIs.

    1. Barrier APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if
       it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy).
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.

    2. Barrier APIs conflict with SyncAll. Avoid mixing them together.

    3. We provide 2 kinds of barrier:
        a. aclshmem_barrier_xxx
            Barrier of all cores. On systems with only HCCS: All operations of all pes of a team on excuting
            stream before the barrier are visiable to all pes of the team after the barrier.
        b. aclshmemx_barrier_xxx_vec
            Barrier of all VEC cores. On systems with only HCCS: All operations of ALL VEC CORES of all pes of a
            team on excuting stream before the barrier are visiable to ALL VEC CORES of all pes of the team after
            the barrier.

        This subtle difference is beneficial to compute-communiction overlapping (usually UNI_DIRECTIONAL dependency),
        and could achieve better performance. Refer to examples/matmul_allreduce for details.

    4. The scalar unit of cube core is not affected by aclshmem_barrier_xxx. Make sure don't use that.
*/

#ifndef ACLSHMEM_DEVICE_CC_H
#define ACLSHMEM_DEVICE_CC_H

#include "host_device/aclshmem_common_types.h"
#include "gm2gm/aclshmem_device_cc.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
 * @brief Set runtime ffts address. Call this at MIX Kernel entry point (if the kernel contains barrier calls).
 *
 * @param config              [config] ffts config, acquired by util_get_ffts_config()
 */
ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config);

/**
 * @fn ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t tid)
 * @brief aclshmem_barrier is a collective synchronization routine over a team. Control returns from aclshmem_barrier
 *        after all PEs in the team have called aclshmem_barrier.
 *        aclshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and
 *        RMA operations, done by any of the PEs in the active set are complete before returning. On systems with
 *        only scale-up network (HCCS), updates are globally visible, whereas on systems with both scale-up network
 *        HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the memory of a given PE are
 *        visible to that PE.
 *        Barrier operations issued on the CPU and the NPU only complete communication operations that were issued
 *        from the CPU and the NPU, respectively. To ensure completion of GPU-side operations from the CPU, using
 *        aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 * @param tid              [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t tid);

/**
 * @fn ACLSHMEM_DEVICE void aclshmem_barrier_all()
 * @brief aclshmem_barrier of all PEs.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all();

/**
 * @brief Similar to aclshmem_barrier except that only vector cores participate. Useful in communication-over-compute
 *        operators. Cube core may call the api but takes no effect.
 *
 * @param tid              [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t tid);

/**
 * @brief aclshmemx_barrier_vec of all PEs.
 *
 * @param tid              [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec();

#ifdef __cplusplus
}
#endif

#endif