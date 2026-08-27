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
#include "unittest/utils/order_kernel.h"

extern "C" ACLSHMEM_GLOBAL void quiet_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);
    __gm__ uint64_t* base = reinterpret_cast<__gm__ uint64_t*>(addr);
    if (rank_id == 0) {
        aclshmemi_store<uint64_t>(base, 0xAA);
        aclshmem_quiet();
        aclshmemi_store<uint64_t>(base + 32, 0xBB);
        dcci_cacheline((__gm__ uint8_t*)(base + 32));
    }

    if (rank_id == 1) {
        uint64_t seen_b;
        __gm__ uint64_t* remote = (__gm__ uint64_t*)aclshmem_ptr(base, 0);
        do {
            dcci_cacheline((__gm__ uint8_t*)(remote + 32));
            seen_b = aclshmemi_load<uint64_t>(remote + 32);
        } while (seen_b != 0xBB);
        dcci_cacheline((__gm__ uint8_t*)(remote));
        uint64_t seen_a = aclshmemi_load<uint64_t>(remote);

        aclshmemi_store<uint64_t>(base + 33, seen_b);
        aclshmemi_store<uint64_t>(base + 34, seen_a);
    }
}

extern "C" ACLSHMEM_GLOBAL void fence_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);
    __gm__ uint64_t* base = reinterpret_cast<__gm__ uint64_t*>(addr);
    if (rank_id == 0) {
        uint64_t a_val = 42, b_val = 84;
        aclshmemi_store<uint64_t>(base, a_val);
        aclshmem_fence();
        aclshmemi_store<uint64_t>(base + 16, b_val);
        dcci_cacheline((__gm__ uint8_t*)(base + 16));
    }

    if (rank_id == 1) {
        uint64_t seen_b;
        __gm__ uint64_t* remote = (__gm__ uint64_t*)aclshmem_ptr(base, 0);
        do {
            dcci_cacheline((__gm__ uint8_t*)(remote + 16));
            seen_b = aclshmemi_load<uint64_t>(remote + 16);
        } while (seen_b != 84);
        dcci_cacheline((__gm__ uint8_t*)remote);
        uint64_t seen_a = aclshmemi_load<uint64_t>(remote);

        aclshmemi_store<uint64_t>(base + 17, seen_b);
        aclshmemi_store<uint64_t>(base + 18, seen_a);
    }
}

extern "C" ACLSHMEM_GLOBAL_VECTOR void quiet_completion(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    // This kernel is used by the SDMA and UDMA completion tests. The MTE
    // default UB reuse case is covered separately by quiet_completion_mte.
    util_set_ffts_config(config);
    __gm__ uint64_t* base = reinterpret_cast<__gm__ uint64_t*>(addr);
    constexpr uint64_t source_words = QUIET_COMPLETION_REPEATS * QUIET_COMPLETION_WORDS;
    uint64_t error_count = 0;

    // An empty quiet must return without waiting for work that was never submitted.
    aclshmem_quiet();

    for (uint64_t round = 0; round < QUIET_COMPLETION_REPEATS; ++round) {
        for (int peer = 0; peer < rank_size; ++peer) {
            if (peer == rank_id) {
                continue;
            }
            __gm__ uint64_t* dst =
                base + source_words +
                (round * static_cast<uint64_t>(rank_size) + static_cast<uint64_t>(peer)) * QUIET_COMPLETION_WORDS;
            __gm__ uint64_t* src = base + round * QUIET_COMPLETION_WORDS;
            aclshmem_getmem_nbi(dst, src, QUIET_COMPLETION_WORDS * sizeof(uint64_t), peer);
        }

        aclshmem_quiet();

        for (int peer = 0; peer < rank_size; ++peer) {
            if (peer == rank_id) {
                continue;
            }
            __gm__ uint64_t* dst =
                base + source_words +
                (round * static_cast<uint64_t>(rank_size) + static_cast<uint64_t>(peer)) * QUIET_COMPLETION_WORDS;
            for (uint64_t i = 0; i < QUIET_COMPLETION_WORDS; ++i) {
                uint64_t expected = quiet_completion_value(round, static_cast<uint64_t>(peer), i);
                if (dst[i] != expected) {
                    ++error_count;
                }
            }
        }
    }

    uint64_t status_offset =
        source_words + QUIET_COMPLETION_REPEATS * static_cast<uint64_t>(rank_size) * QUIET_COMPLETION_WORDS;
    base[status_offset] = error_count;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(base + status_offset));
}

// Standard MTE NBI uses the state-owned default UB. Serialize its reuse with
// the configured MTE3->MTE2 event, then keep one quiet for the whole batch.
extern "C" ACLSHMEM_GLOBAL_VECTOR void quiet_completion_mte(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);
    __gm__ uint64_t* base = reinterpret_cast<__gm__ uint64_t*>(addr);
    constexpr uint64_t source_words = QUIET_COMPLETION_REPEATS * QUIET_COMPLETION_WORDS;
    AscendC::TEventID event_id = static_cast<AscendC::TEventID>(aclshmemi_get_state()->mte_config.sync_id);
    uint64_t error_count = 0;

    // An empty quiet must return without waiting for work that was never submitted.
    aclshmem_quiet();
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

    for (uint64_t round = 0; round < QUIET_COMPLETION_REPEATS; ++round) {
        for (int peer = 0; peer < rank_size; ++peer) {
            if (peer == rank_id) {
                continue;
            }
            __gm__ uint64_t* dst =
                base + source_words +
                (round * static_cast<uint64_t>(rank_size) + static_cast<uint64_t>(peer)) * QUIET_COMPLETION_WORDS;
            __gm__ uint64_t* src = base + round * QUIET_COMPLETION_WORDS;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
            aclshmem_getmem_nbi(dst, src, QUIET_COMPLETION_WORDS * sizeof(uint64_t), peer);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        }

        aclshmem_quiet();

        for (int peer = 0; peer < rank_size; ++peer) {
            if (peer == rank_id) {
                continue;
            }
            __gm__ uint64_t* dst =
                base + source_words +
                (round * static_cast<uint64_t>(rank_size) + static_cast<uint64_t>(peer)) * QUIET_COMPLETION_WORDS;
            for (uint64_t i = 0; i < QUIET_COMPLETION_WORDS; ++i) {
                uint64_t expected = quiet_completion_value(round, static_cast<uint64_t>(peer), i);
                if (dst[i] != expected) {
                    ++error_count;
                }
            }
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);

    uint64_t status_offset =
        source_words + QUIET_COMPLETION_REPEATS * static_cast<uint64_t>(rank_size) * QUIET_COMPLETION_WORDS;
    base[status_offset] = error_count;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(base + status_offset));
}

// This kernel deliberately uses the RoCE-specific APIs.  Keep the generic
// quiet_completion kernel above unchanged so the MTE/SDMA/UDMA tests continue
// to exercise normal transport dispatch.
extern "C" ACLSHMEM_GLOBAL_VECTOR void quiet_completion_rdma(uint64_t config, GM_ADDR addr, int rank_id, int rank_size)
{
    util_set_ffts_config(config);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE_64 * 2);
    AscendC::LocalTensor<uint8_t> roce_ub = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE_64 * 2, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

    __gm__ uint64_t* base = reinterpret_cast<__gm__ uint64_t*>(addr);
    __gm__ aclshmem_device_host_state_t* state = aclshmemi_get_state();
    constexpr uint64_t source_words = QUIET_COMPLETION_REPEATS * QUIET_COMPLETION_WORDS;
    // Match the state-owned sync ID used by generic RoCE quiet.
    const uint32_t sync_id = state->rdma_config.sync_id;
    __ubuf__ uint64_t* roce_ub_addr = reinterpret_cast<__ubuf__ uint64_t*>(roce_ub.GetPhyAddr());
    uint64_t error_count = 0;

    // aclshmemx_roce_quiet is per peer; with no NBI submitted it must return for every reachable peer.
    for (int peer = 0; peer < rank_size; ++peer) {
        if (peer == rank_id) {
            continue;
        }
        if ((state->topo_list[peer] & ACLSHMEM_TRANSPORT_ROCE) == 0) {
            ++error_count;
            continue;
        }
        aclshmemx_roce_quiet(peer, roce_ub_addr, sync_id);
    }
    aclshmem_quiet();

    for (uint64_t round = 0; round < QUIET_COMPLETION_REPEATS; ++round) {
        for (int peer = 0; peer < rank_size; ++peer) {
            if (peer == rank_id) {
                continue;
            }
            if ((state->topo_list[peer] & ACLSHMEM_TRANSPORT_ROCE) == 0) {
                ++error_count;
                continue;
            }
            __gm__ uint64_t* dst =
                base + source_words +
                (round * static_cast<uint64_t>(rank_size) + static_cast<uint64_t>(peer)) * QUIET_COMPLETION_WORDS;
            __gm__ uint64_t* src = base + round * QUIET_COMPLETION_WORDS;
            // Force the RoCE submission path even when this peer also has MTE
            // reachability. Unlike aclshmem_getmem_nbi, elem_size is counted in
            // uint64_t elements.
            aclshmemx_roce_get_nbi(dst, src, roce_ub_addr, QUIET_COMPLETION_WORDS, peer, sync_id);
        }

        // Exercise generic quiet: it must find the forced RoCE work through the
        // transport mask and perform both RoCE completion and cache invalidation.
        aclshmem_quiet();

        for (int peer = 0; peer < rank_size; ++peer) {
            if (peer == rank_id) {
                continue;
            }
            __gm__ uint64_t* dst =
                base + source_words +
                (round * static_cast<uint64_t>(rank_size) + static_cast<uint64_t>(peer)) * QUIET_COMPLETION_WORDS;
            for (uint64_t i = 0; i < QUIET_COMPLETION_WORDS; ++i) {
                uint64_t expected = quiet_completion_value(round, static_cast<uint64_t>(peer), i);
                if (dst[i] != expected) {
                    ++error_count;
                }
            }
        }
    }

    uint64_t status_offset =
        source_words + QUIET_COMPLETION_REPEATS * static_cast<uint64_t>(rank_size) * QUIET_COMPLETION_WORDS;
    base[status_offset] = error_count;
    dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(base + status_offset));
}

void quiet_order_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks)
{
    quiet_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}

void fence_order_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks)
{
    fence_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}

void quiet_completion_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks)
{
    quiet_completion<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}

void quiet_completion_mte_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks)
{
    quiet_completion_mte<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}

void quiet_completion_rdma_do(void* stream, uint64_t config, uint8_t* addr, int32_t rank_id, int32_t n_ranks)
{
    quiet_completion_rdma<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}
