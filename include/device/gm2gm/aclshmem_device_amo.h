/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_DEVICE_ATOMIC_H
#define ACLSHMEM_DEVICE_ATOMIC_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/aclshmem_device_mte.h"
#include "gm2gm/aclshmem_device_amo.hpp"

#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE, ATOMIC_TYPE)                                                         \
    /**                                                                                                            \
     * @brief Asynchronous interface. Perform contiguous data atomic add opeartion                                 \
              on symmetric memory from the specified PE to address on the local PE.                                \
              We use scalar instructions to implement single element atomic add. Therefore, both                   \
     *        vector and cube cores can execute atomic add operation.                                              \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param value             [in] Value atomic add to destination.                                              \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe);

ACLSHMEM_TYPE_FUNC_ATOMIC_INT(ACLSHMEM_ATOMIC_ADD_TYPENAME);

#define ACLSHMEM_ATOMIC_ADD_TYPENAME_FLOAT(NAME, TYPE, ATOMIC_TYPE)                                                   \
    /**                                                                                                            \
     * @brief Asynchronous interface. Perform contiguous data atomic add opeartion on                              \
              symmetric memory from the specified PE to address on the local PE.                                   \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param value             [in] Value atomic add to destination.                                              \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe);

ACLSHMEM_TYPE_FUNC_ATOMIC_FLOAT(ACLSHMEM_ATOMIC_ADD_TYPENAME_FLOAT);
#endif