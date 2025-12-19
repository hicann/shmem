/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_DEVICE_CC_HPP
#define ACLSHMEM_DEVICE_CC_HPP

#include "host_device/aclshmem_common_types.h"
#include "aclshmemi_device_cc.h"
#include "gm2gm/engine/aclshmemi_device_rdma.h"

ACLSHMEM_DEVICE void aclshmemi_barrier_cross_host(aclshmemx_team_t *team)
{
    if (AscendC::GetBlockIdx() != 0)
        return;

    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = aclshmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t *)sync_counter) + 1;
    aclshmemi_store((__gm__ int32_t *)sync_counter, count);
    dcci_cacheline((__gm__ uint8_t *)sync_counter);
    while (shift < size) {
        int pre_pe_in_team = (my_pe_in_team - shift + size) % size;
        int next_pe_in_team = (my_pe_in_team + shift) % size;

        int pre_pe = start + pre_pe_in_team * stride;
        int next_pe = start + next_pe_in_team * stride;

        // signal next pe
        aclshmemi_highlevel_signal_set((__gm__ int32_t *)(sync_array + my_pe), (__gm__ int32_t *)sync_counter, next_pe);

        // wait pre pe
        aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_array + pre_pe), count);

        shift *= SHIFT_MULTIPLIER;
    }
}

ACLSHMEM_DEVICE void aclshmemi_handle(aclshmem_team_t tid)
{
    aclshmemx_team_t *team = aclshmemi_get_state()->team_pools[tid];

    int mype = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        // not in this team
        return;
    }

    AscendC::LocalTensor<uint32_t> ub_tensor_32;
    ub_tensor_32.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_32.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR);
    ub_tensor_32.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> ub_tensor_64;
    ub_tensor_64.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor_64.address_.bufferAddr = reinterpret_cast<uint64_t>(ACLSHMEM_INTERNAL_UB_BUF_START_ADDR
                                                                        + UB_ALIGN_SIZE);
    ub_tensor_64.address_.dataLen = UB_ALIGN_SIZE;

    for (int i = 0; i < size; i++) {
        int peer = start + i * stride;
        if (peer == mype) {
            continue;
        }
        aclshmemi_roce_quiet(peer, 0, ub_tensor_64, ub_tensor_32);
    }

    if ASCEND_IS_AIV {
        aclshmemi_barrier_cross_host(team);
    }
}

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_core_sync_array()
{
    return (__gm__ aclshmemi_sync_bit *)aclshmemi_get_state()->core_sync_pool;
}

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_core_sync_counter()
{
    return (__gm__ aclshmemi_sync_bit *)aclshmemi_get_state()->core_sync_counter;
}

ACLSHMEM_DEVICE void aclshmemi_barrier_core_soft()
{
#ifdef __DAV_C220_VEC__
    auto sync_array = aclshmemi_get_core_sync_array();
    auto sync_counter = aclshmemi_get_core_sync_counter();

    int idx = AscendC::GetBlockIdx();
    int size = AscendC::GetBlockNum();
    int count = aclshmemi_load((__gm__ int32_t *)(sync_counter)) + 1;

    int shift = 1;
    int offset = 0;
    while (shift < size) {
        int next = (idx + shift) % size;

        aclshmemi_signal_set((__gm__ int32_t *)(sync_array + next * ACLSHMEM_LOG_MAX_AIV_PER_NPU + offset), count);
        aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_array +
            idx * ACLSHMEM_LOG_MAX_AIV_PER_NPU + offset), count);

        shift *= SHIFT_MULTIPLIER;
        offset++;
    }

    aclshmemi_store((__gm__ int32_t *)(sync_counter), count);
#endif
}

template<bool is_aiv_only>
ACLSHMEM_DEVICE void aclshmemi_barrier_core()
{
#ifdef __CCE_AICORE_ENABLE_MIX__
    AscendC::SyncAll<is_aiv_only>();
#else
    aclshmemi_barrier_core_soft();
#endif
}

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_team_sync_array(aclshmem_team_t team_idx)
{
    uint64_t addr = (uint64_t) aclshmemi_get_state()->sync_pool;
    addr += team_idx * SYNC_ARRAY_SIZE;
    return (__gm__ aclshmemi_sync_bit *) addr;
}

ACLSHMEM_DEVICE
__gm__ aclshmemi_sync_bit *aclshmemi_get_team_sync_counter(aclshmem_team_t team_idx)
{
    uint64_t addr = (uint64_t) aclshmemi_get_state()->sync_counter;
    addr += team_idx * SYNC_COUNTER_SIZE;
    return (__gm__ aclshmemi_sync_bit *) addr;
}

ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v1(aclshmemx_team_t *team)
{
    if (AscendC::GetBlockIdx() != 0)
        return;

    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = aclshmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t *)sync_counter) + 1;

    int32_t offset = 0;
    const int multiplier = 2;
    while (shift < size) {
        int next_pe_in_team = (my_pe_in_team + shift) % size;
        int next_pe = start + next_pe_in_team * stride;

        // signal next pe
        aclshmemi_signal_set((__gm__ int32_t *)(sync_array + offset), next_pe, count);

        // wait pre pe
        aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_array + offset), count);

        shift *= multiplier;
        offset++;
    }

    aclshmemi_store((__gm__ int32_t *)sync_counter, count);
}

ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v2(aclshmemx_team_t *team)
{
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = aclshmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int k = ACLSHMEM_BARRIER_TG_DISSEM_KVAL;
    k = k < size ? k : size;
    k = k < vec_size ? k : vec_size;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t *)sync_counter) + 1;

    int32_t offset = 0;
    while (shift < size) {
        for (int i = vec_id + 1; i < k; i += vec_size) {
            int next_pe_in_team = (my_pe_in_team + i * shift) % size;
            int next_pe = start + next_pe_in_team * stride;

            // signal next pe
            aclshmemi_signal_set((__gm__ int32_t *)(sync_array + offset + i), next_pe, count);
        }

        for (int i = vec_id + 1; i < k; i += vec_size) {
            // wait pre pe
            aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)(sync_array + offset + i), count);
        }

        shift *= k;
        offset += k;
    }

    aclshmemi_store((__gm__ int32_t *)sync_counter, count);
}

ACLSHMEM_DEVICE void aclshmemi_barrier_npu_v3(aclshmemx_team_t *team)
{
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    int my_pe = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = aclshmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = aclshmemi_get_team_sync_counter(team->team_idx);

    int k = ACLSHMEM_BARRIER_TG_DISSEM_KVAL;
    k = k < size ? k : size;
    k = k < vec_size ? k : vec_size;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = aclshmemi_load((__gm__ int32_t *)sync_counter) + 1;

    for (int i = vec_id; i < size; i += k) {
        if (i == my_pe_in_team) {
            // write local
            aclshmemi_signal_set((__gm__ int32_t *)sync_array, count);
        } else {
            // read remote
            int remote_pe = start + i * stride;
            aclshmemi_signal_wait_until_eq_for_barrier((__gm__ int32_t *)aclshmem_ptr(sync_array, remote_pe), count);
        }
    }

    aclshmemi_store((__gm__ int32_t *)sync_counter, count);
}

template<bool is_aiv_only>
ACLSHMEM_DEVICE void aclshmemi_barrier(aclshmem_team_t tid)
{
    aclshmemx_team_t *team = aclshmemi_get_state()->team_pools[tid];

    int mype = aclshmemi_get_state()->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        // not in this team
        return;
    }

    aclshmemi_barrier_core<is_aiv_only>();

    if ASCEND_IS_AIV {
        aclshmemi_barrier_npu_v3(team);
    }

    aclshmemi_barrier_core<is_aiv_only>();
}

ACLSHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t * addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dcci_cachelines(__gm__ uint8_t* addr, uint64_t length)
{
    __gm__ uint8_t* start = (__gm__ uint8_t*)((uint64_t)addr / ACLSHMEM_DATA_CACHE_LINE_SIZE
        * ACLSHMEM_DATA_CACHE_LINE_SIZE);
    __gm__ uint8_t* end = (__gm__ uint8_t*)(((uint64_t)addr + length) / ACLSHMEM_DATA_CACHE_LINE_SIZE
        * ACLSHMEM_DATA_CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint64_t i = 0; i <= end - start; i+= ACLSHMEM_DATA_CACHE_LINE_SIZE) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<uint8_t,
            AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[i]);
        __asm__ __volatile__("");
    }
}

ACLSHMEM_DEVICE void dcci_entire_cache()
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    
    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dcci_atomic()
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;

    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_ATOMIC>(global);
    __asm__ __volatile__("");
}

ACLSHMEM_DEVICE void dsb_all()
{
    using namespace AscendC;
    
    DataSyncBarrier<MemDsbT::ALL>();
}

#ifdef __cplusplus
extern "C" {
#endif

ACLSHMEM_DEVICE void util_set_ffts_config(uint64_t config)
{
    AscendC::SetSyncBaseAddr(config);
}

ACLSHMEM_DEVICE void aclshmem_barrier(aclshmem_team_t tid)
{
    aclshmemi_barrier<false>(tid);
}

ACLSHMEM_DEVICE void aclshmem_barrier_all()
{
    aclshmem_barrier(ACLSHMEM_TEAM_WORLD);
}

ACLSHMEM_DEVICE void aclshmemx_barrier_vec(aclshmem_team_t tid)
{
    aclshmemi_barrier<true>(tid);
}

ACLSHMEM_DEVICE void aclshmemx_barrier_all_vec()
{
    aclshmemx_barrier_vec(ACLSHMEM_TEAM_WORLD);
}

#ifdef __cplusplus
}
#endif

#endif