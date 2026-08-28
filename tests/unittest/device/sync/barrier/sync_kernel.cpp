/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"
#include "gm2gm/shmemi_device_cc.h"
#include "shmemi_device_common.h"
#include "unittest/utils/sync_kernel.h"

namespace {
constexpr uint64_t kSyncStressDelayCycles = 200;
} // namespace

extern "C" ACLSHMEM_GLOBAL void sync_increase(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
    // The scalar unit of a cube core is not synchronized by the collective.
    aclshmem_sync_all();
    aclshmem_sync_all();
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t*)addr);

    aclshmem_sync_all();
    GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
    aclshmemi_store((__gm__ uint64_t*)remote, val + 1);
    aclshmem_sync_all();
#endif
}

extern "C" ACLSHMEM_GLOBAL void sync_increase_vec(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t*)addr);

    aclshmemx_sync_vec_all();
    GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
    aclshmemi_store((__gm__ uint64_t*)remote, val + 1);
    aclshmemx_sync_vec_all();
#endif
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void sync_increase_single_aiv(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t*)addr);

    aclshmemx_sync_vec_all();
    GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
    aclshmemi_store((__gm__ uint64_t*)remote, val + 1);
    aclshmemx_sync_vec_all();
#endif
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void sync_v4_single_aiv(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t*)addr);

    aclshmemi_sync_npu_v4<true>(ACLSHMEM_TEAM_WORLD);
    GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
    aclshmemi_store((__gm__ uint64_t*)remote, val + 1);
    aclshmemi_sync_npu_v4<true>(ACLSHMEM_TEAM_WORLD);
#endif
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void sync_stress_vec(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    const uint32_t vec_id = AscendC::GetBlockIdx();
    for (uint32_t iteration = 0; iteration < ACLSHMEM_SYNC_STRESS_ITERATIONS; ++iteration) {
        const uint64_t delay_cycles =
            ((vec_id + static_cast<uint32_t>(rank_id) + iteration) & 0x7U) * kSyncStressDelayCycles;
        const uint64_t delay_start = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - delay_start < delay_cycles) {
        }
        aclshmemx_sync_vec_all();
    }

    if (vec_id == 0U) {
        GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
        aclshmemi_store((__gm__ uint64_t*)remote, static_cast<uint64_t>(ACLSHMEM_SYNC_STRESS_ITERATIONS));
    }
    aclshmemx_sync_vec_all();
#endif
}

extern "C" ACLSHMEM_GLOBAL void sync_core_soft_mixed_stress(
    uint64_t config, GM_ADDR observed_aiv_count, GM_ADDR completion, int rank_id)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    const uint32_t vec_id = AscendC::GetBlockIdx();
    const uint32_t vec_count = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    for (uint32_t iteration = 0; iteration < ACLSHMEM_SYNC_STRESS_ITERATIONS; ++iteration) {
        const uint64_t delay_cycles =
            ((vec_id + static_cast<uint32_t>(rank_id) + iteration) & 0x7U) * kSyncStressDelayCycles;
        const uint64_t delay_start = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - delay_start < delay_cycles) {
        }
        aclshmemi_sync_core_soft();
    }

    __gm__ uint32_t* completion_slot =
        (__gm__ uint32_t*)(completion + static_cast<uint64_t>(vec_id) * ACLSHMEM_SYNCBIT_SIZE);
    aclshmemi_store(completion_slot, 1U);
    dcci_cacheline((__gm__ uint8_t*)completion_slot);
    if (vec_id == 0U) {
        aclshmemi_store((__gm__ uint32_t*)observed_aiv_count, vec_count);
        dcci_cacheline((__gm__ uint8_t*)observed_aiv_count);
    }
#endif
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void sync_v4_stress_vec(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    const uint32_t vec_id = AscendC::GetBlockIdx();
    for (uint32_t iteration = 0; iteration < ACLSHMEM_SYNC_STRESS_ITERATIONS; ++iteration) {
        const uint64_t delay_cycles =
            ((vec_id + static_cast<uint32_t>(rank_id) + iteration) & 0x7U) * kSyncStressDelayCycles;
        const uint64_t delay_start = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - delay_start < delay_cycles) {
        }
        aclshmemi_sync_npu_v4<true>(ACLSHMEM_TEAM_WORLD);
    }

    if (vec_id == 0U) {
        GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 1) % rank_size);
        aclshmemi_store((__gm__ uint64_t*)remote, static_cast<uint64_t>(ACLSHMEM_SYNC_STRESS_ITERATIONS));
    }
    aclshmemi_sync_npu_v4<true>(ACLSHMEM_TEAM_WORLD);
#endif
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void sync_singleton_team(uint64_t config, GM_ADDR addr, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    aclshmemi_sync_npu_v4<true>(team_id);
    aclshmemi_store((__gm__ uint64_t*)addr, static_cast<uint64_t>(1));
    aclshmemi_sync_npu_v4<true>(team_id);
#endif
}

extern "C" ACLSHMEM_GLOBAL void sync_increase_odd_team(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
    // The scalar unit of a cube core is not synchronized by the collective.
    aclshmem_sync_all();
    aclshmem_sync_all();
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t*)addr);

    aclshmem_sync(team_id);
    if (rank_id & 1) {
        GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 2) % rank_size);
        aclshmemi_store((__gm__ uint64_t*)remote, val + 1);
    }
    aclshmem_sync(team_id);
#endif
}

extern "C" ACLSHMEM_GLOBAL void sync_increase_vec_odd_team(
    uint64_t config, GM_ADDR addr, int rank_id, int rank_size, aclshmem_team_t team_id)
{
    util_set_ffts_config(config);

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    uint64_t val = aclshmemi_load((__gm__ uint64_t*)addr);

    aclshmemx_sync_vec(team_id);
    if (rank_id & 1) {
        GM_ADDR remote = (GM_ADDR)aclshmem_ptr(addr, (rank_id + 2) % rank_size);
        aclshmemi_store((__gm__ uint64_t*)remote, val + 1);
    }
    aclshmemx_sync_vec(team_id);
#endif
}

void sync_increase_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size)
{
    sync_increase<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void sync_increase_vec_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size)
{
    sync_increase_vec<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void sync_increase_single_aiv_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size)
{
    sync_increase_single_aiv<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void sync_v4_single_aiv_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size)
{
    sync_v4_single_aiv<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void sync_stress_vec_do(void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size)
{
    sync_stress_vec<<<ACLSHMEM_SYNC_STRESS_AIV_COUNT, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void sync_core_soft_mixed_stress_do(
    uint32_t block_num, void* stream, uint64_t config, uint8_t* observed_aiv_count, uint8_t* completion, int rank_id)
{
    sync_core_soft_mixed_stress<<<block_num, nullptr, stream>>>(config, observed_aiv_count, completion, rank_id);
}

void sync_v4_stress_vec_do(uint32_t block_num, void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size)
{
    sync_v4_stress_vec<<<block_num, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void sync_singleton_team_do(void* stream, uint64_t config, uint8_t* addr, aclshmem_team_t team_id)
{
    sync_singleton_team<<<1, nullptr, stream>>>(config, addr, team_id);
}

void sync_increase_do_odd_team(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, aclshmem_team_t team_id)
{
    sync_increase_odd_team<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size, team_id);
}

void sync_increase_vec_do_odd_team(
    void* stream, uint64_t config, uint8_t* addr, int rank_id, int rank_size, aclshmem_team_t team_id)
{
    sync_increase_vec_odd_team<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size, team_id);
}
