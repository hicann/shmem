/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_DEVICE_P2P_SYNC_H
#define SHMEM_DEVICE_P2P_SYNC_H

#include "host_device/common_types.h"
#include "internal/device/sync/shmemi_device_p2p.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The shmemx_signal_op operation updates sig_addr with signal using operation sig_op on the specified PE.
 *        This operation can be used together with shmem_signal_wait_until for efficient point-to-point synchronization.
 * WARNING: Atomicity NOT Guaranteed.
 *
 * @param sig_addr              [in] Symmetric address of the signal word to be updated.
 * @param signal                [in] The value used to update sig_addr.
 * @param sig_op                [in] Operation used to update sig_addr with signal. Supported operations:
 *                                   SHMEM_SIGNAL_SET/SHMEM_SIGNAL_ADD
 * @param pe                    [in] PE number of the remote PE.
 */
SHMEM_DEVICE void shmemx_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe)
{
    shmemix_signal_op(sig_addr, signal, sig_op, pe);
}

/**
 * @brief This routine can be used to implement point-to-point synchronization between PEs or between threads within
 *        the same PE. A call to this routine blocks until the value of sig_addr at the calling PE satisfies the wait
 *        condition specified by the comparison operator, cmp, and comparison value, cmp_val.
 *
 * @param sig_addr              [in] Local address of the source signal variable.
 * @param cmp                   [in] The comparison operator that compares sig_addr with cmp_val. Supported operators:
 *                                   SHMEM_CMP_EQ/SHMEM_CMP_NE/SHMEM_CMP_GT/SHMEM_CMP_GE/SHMEM_CMP_LT/SHMEM_CMP_LE.
 * @param cmp_val               [in] The value against which the object pointed to by sig_addr will be compared.
 * @return Return the contents of the signal data object, sig_addr, at the calling PE that satisfies the wait condition.
 */
SHMEM_DEVICE int32_t shmem_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    return shmemi_signal_wait_until(sig_addr, cmp, cmp_val);
}

#ifdef __cplusplus
}
#endif

#endif