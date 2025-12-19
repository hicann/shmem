/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_DEVICE_UB2GM_MTE_H
#define ACLSHMEM_DEVICE_UB2GM_MTE_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/aclshmem_device_mte.h"
#include "device/aclshmem_def.h"
#include "ub2gm/mte/aclshmem_device_mte.hpp"

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE
 *        to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */

#endif