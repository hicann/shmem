/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_DEVICE_TEAM_HPP
#define ACLSHMEM_DEVICE_TEAM_HPP

#include "host_device/aclshmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE int aclshmem_my_pe(void)
{
    return aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
}

ACLSHMEM_DEVICE int aclshmem_n_pes(void)
{
    return aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->size;
}

ACLSHMEM_DEVICE int aclshmem_team_my_pe(aclshmem_team_t team)
{
    if (team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    } else {
        aclshmemx_team_t *src_team_ptr = aclshmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->mype : -1);
    }
}

ACLSHMEM_DEVICE int aclshmem_team_n_pes(aclshmem_team_t team)
{
    if (team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    } else {
        aclshmemx_team_t *src_team_ptr = aclshmemi_get_state()->team_pools[team];
        return (src_team_ptr != nullptr ? src_team_ptr->size : -1);
    }
}

ACLSHMEM_DEVICE int aclshmem_team_translate_pe(aclshmem_team_t src_team, int src_pe, aclshmem_team_t dest_team)
{
    if (src_team == ACLSHMEM_TEAM_INVALID || dest_team == ACLSHMEM_TEAM_INVALID) {
        return -1;
    }
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();

    aclshmemx_team_t *src_team_ptr = device_state->team_pools[src_team];
    aclshmemx_team_t *dest_team_ptr = device_state->team_pools[dest_team];

    if (src_pe > src_team_ptr->size) {
        return -1;
    }

    int global_pe = src_team_ptr->start + src_pe * src_team_ptr->stride;
    int pe_start = dest_team_ptr->start;
    int pe_stride = dest_team_ptr->stride;
    int pe_size = dest_team_ptr->size;

    int n = (global_pe - pe_start) / pe_stride;
    if (global_pe < pe_start || (global_pe - pe_start) % pe_stride || n >= pe_size) {
        return -1;
    }

    return n;
}

#ifdef __cplusplus
}
#endif

#endif