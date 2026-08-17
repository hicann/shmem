/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TP_ALLREDUCE_UDMA_KERNEL_H
#define TP_ALLREDUCE_UDMA_KERNEL_H

#include <cstdint>

#include "shmem.h"

enum class TpAllReduceUdmaMode : int32_t {
    BASELINE = 0,
    TAILCUT = 1,
};

enum class TpAllReduceUdmaStage : int32_t {
    REDUCE_SCATTER = 0,
    LOCAL_REDUCE_ALLGATHER = 1,
};

void launch_tp_allreduce_udma_int32_t(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, int32_t mode, int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight);

void launch_tp_allreduce_udma_float16_t(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, int32_t mode, int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight);

#endif // TP_ALLREDUCE_UDMA_KERNEL_H
