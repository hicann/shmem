/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */

// clang-format off
/*
 * WARNING: Restrictions of device collective synchronization APIs.
 *
 * 1. All-core synchronization APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC
 *    or CUBE if it lacks effective cube instructions (for example, Mmad) or vector instructions (for example,
 *    DataCopy). Compiler support is required to remove this restriction; otherwise insert effective Mmad/DataCopy
 *    calls.
 *
 * 2. ACLSHMEM inter-PE synchronization conflicts with SyncAll. Avoid mixing them in the same kernel.
 *
 * 3. We provide two participant scopes:
 *    a. aclshmem_barrier_xxx: all cores participate.
 *    b. aclshmemx_sync_vec / aclshmemx_sync_vec_all: only VEC cores participate. On systems using HCCS or UB,
 *       memory stores issued by participating VEC cores before the sync are visible to participating VEC cores after
 *       the sync. The sync does not complete remote updates issued through ACLSHMEM communication APIs.
 *
 * 4. The scalar unit of a cube core is not synchronized by these APIs. Do not rely on it across the collective.
 */
// clang-format on

/*!
 * \file shmem_device_cc.h
 * \brief shmem device Collective Communication APIs
 */
#ifndef _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
#define _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_

#include "host_device/shmem_common_types.h"

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
#define shmemx_set_ffts_config util_set_ffts_config

/**
 * @brief aclshmem_barrier is a collective synchronization routine over a team. Control returns from aclshmem_barrier
 *        after all PEs in the team have called aclshmem_barrier.
 *        aclshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and
 *        RMA operations, done by any of the PEs in the active set are complete before returning. On systems with
 *        only scale-up network (HCCS), updates are globally visible, whereas on systems with both scale-up network
 *        HCCS and scale-out network (RDMA), ACLSHMEM only guarantees that updates to the memory of a given PE are
 *        visible to that PE.
 *        Barrier operations issued on the CPU and the NPU only complete communication operations that were issued
 *        from the CPU and the NPU, respectively. To ensure completion of NPU-side operations from the CPU, using
 *        aclrtSynchronizeStream/aclrtDeviceSynchronize or stream-based API.
 *
 * @param team              [in] team to do barrier
 */
ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t team);
#define shmem_barrier aclshmem_barrier

/**
 * @brief aclshmem_barrier of all PEs.
 */
ACLSHMEM_DEVICE void aclshmem_barrier_all(void);
#define shmem_barrier_all aclshmem_barrier_all

/**
 * @brief Collective synchronization over a team in which only vector cores participate. This routine does not
 * ensure
 *        completion of remote memory updates issued through ACLSHMEM communication APIs. Cube cores may call
 * the API
 *        but take no effect.
 *
 * @param team              [in] team to synchronize
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec(aclshmem_team_t team);

/**
 * @brief aclshmemx_sync_vec over ACLSHMEM_TEAM_WORLD.
 */
ACLSHMEM_DEVICE void aclshmemx_sync_vec_all(void);

/**
 * @brief Compatibility wrapper for aclshmemx_sync_vec.
 *
 * @param team              [in] team to synchronize
 */
[[deprecated("aclshmemx_barrier_vec is deprecated, please use aclshmemx_sync_vec instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t team);
#define shmemx_barrier_vec aclshmemx_barrier_vec

/**
 * @brief Compatibility wrapper for aclshmemx_sync_vec_all.
 */
[[deprecated("aclshmemx_barrier_all_vec is deprecated, please use aclshmemx_sync_vec_all instead.")]]
ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec(void);
#define shmemx_barrier_all_vec aclshmemx_barrier_all_vec

/**
 * @brief Similar to aclshmem_barrier. In contrast with the aclshmem_barrier routine, aclshmem_sync only ensures
 *        completion and visibility of previously issued memory stores and does not ensure completion of remote memory
 *        updates issued via ACLSHMEM routines.
 *
 * @param team           [in] team to synchronize
 */
ACLSHMEM_DEVICE void aclshmem_sync(aclshmem_team_t team);

/**
 * @brief aclshmem_sync_all of all PEs.
 *
 */
ACLSHMEM_DEVICE void aclshmem_sync_all(void);

#ifdef __cplusplus
}
#endif

#include "gm2gm/shmemi_device_cc.h"

#endif // _DEVICE_GM2GM_ACLSHMEM_DEVICE_CC_H_
