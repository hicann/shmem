/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SDMA_PERFTEST_KERNEL_H
#define SDMA_PERFTEST_KERNEL_H

#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"
#include "perftest_common_types.h"

template <typename T>
__aicore__ inline void sdma_perf_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, GM_ADDR timing_out_gva, int elements, int32_t frame_id,
    int64_t prof_pe, int loop_count, int ub_size_kb, bool bidirectional, bool is_put, uint32_t qp_num, int metric,
    int batch)
{
    // Configure the FFTS base address before any ACLSHMEM collective sync usage (see allgather/rdma_perftest).
    util_set_ffts_config(fftsAddr);
    if ASCEND_IS_AIV {
        int64_t pe = aclshmem_my_pe();
        int peer_pe = (pe + 1) % aclshmem_n_pes();
        int aiv_num = static_cast<int>(qp_num);
        int aiv_idx = AscendC::GetBlockIdx();

        // Cross-PE timing slots, rdma_perftest style: region 0 holds PE0's per-AIV
        // cycles, region 1 holds PE1's (this example is a 2-PE tool, see rdma_perftest).
        __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);
        const int timing_region = (pe == 0) ? 0 : 1;
        __gm__ int64_t* timing_slot =
            (timing_out != nullptr) ? timing_out + timing_region * ACLSHMEM_MAX_AIV_PER_NPU + aiv_idx : nullptr;

        // Bidirectional tests run on every PE; unilateral tests only on the prof PE.
        // Idle AIVs (aiv_idx >= qp_num / no data) skip the SDMA traffic but must still
        // reach the collective syncs below, otherwise sync_vec_all deadlocks.
        bool is_runner = bidirectional || pe == prof_pe;
        bool is_worker = is_runner && aiv_idx < aiv_num;

        constexpr uint32_t ub_offset = 1024;
        constexpr uint32_t ub_size = 64;
        __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(static_cast<uint64_t>(ub_offset));
        __gm__ T* dst = reinterpret_cast<__gm__ T*>(dst_gva);
        __gm__ T* src = reinterpret_cast<__gm__ T*>(src_gva);
        // Every worker AIV/QP transfers a complete data_size message. Each AIV gets a
        // private region in the symmetric buffers so requests do not overlap. Keep the
        // offset in int64_t because elements * aiv_idx can exceed int32.
        const int local_elements = is_worker ? elements : 0;
        const int64_t offset = static_cast<int64_t>(elements) * aiv_idx;
        bool has_traffic = is_worker && local_elements > 0;

        // rdma-style cross-PE alignment: both PEs enter the test (and therefore the
        // bidirectional traffic window) at the same time.
        aclshmemx_sync_vec_all();

        if (has_traffic) {
            AscendC::PipeBarrier<PIPE_ALL>();

            // Warmup: submit PERFTEST_WARMUP_ITERS nbi ops and drain them with one quiet.
            for (int i = 0; i < perftest::PERFTEST_WARMUP_ITERS; ++i) {
                if (is_put) {
                    aclshmemx_sdma_qp_put_nbi(
                        dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements), peer_pe,
                        aiv_idx, EVENT_ID0);
                } else {
                    aclshmemx_sdma_qp_get_nbi(
                        dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements), peer_pe,
                        aiv_idx, EVENT_ID0);
                }
            }
            aclshmemx_sdma_qp_quiet(ub_ptr, ub_size, aiv_idx, EVENT_ID0);
        }

        // Timing alignment: every VEC core of every PE joins this ACLSHMEM collective sync,
        // so all workers enter the measured window at the same instant and the max
        // per-block window equals the wall time of the slowest worker. SyncAll must NOT be
        // used here: it conflicts with ACLSHMEM inter-PE syncs in the same kernel
        // (see shmem_device_cc.h restriction 2).
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemx_sync_vec_all();

        if (has_traffic) {
            const int batch_size = (batch <= 0 || batch > loop_count) ? loop_count : batch;

            if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
                // Latency: time loop_count nbi submits in a single window, quiet outside.
                // Divide the collected window time by loop_count to get the per-op latency.
                SHMEMI_PROF_START(frame_id);
                const int64_t lat_time_start = AscendC::GetSystemCycle();
                for (int i = 0; i < loop_count; ++i) {
                    if (is_put) {
                        aclshmemx_sdma_qp_put_nbi(
                            dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements), peer_pe,
                            aiv_idx, EVENT_ID0);
                    } else {
                        aclshmemx_sdma_qp_get_nbi(
                            dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements), peer_pe,
                            aiv_idx, EVENT_ID0);
                    }
                }
                const int64_t lat_time_end = AscendC::GetSystemCycle();
                SHMEMI_PROF_END(frame_id);
                aclshmemx_sdma_qp_quiet(ub_ptr, ub_size, aiv_idx, EVENT_ID0);
                if (timing_slot != nullptr) {
                    *timing_slot = lat_time_end - lat_time_start;
                }
            } else {
                // Bandwidth: submit nbis in groups of batch_size and quiet at each group
                // boundary. batch_size == loop_count -> one trailing quiet (full
                // async); batch_size == 1 -> synchronous; mid sizes split into full groups
                // plus one extra quiet for the tail.
                const int full_groups = loop_count / batch_size;
                const int remainder_iters = loop_count - full_groups * batch_size;
                SHMEMI_PROF_START(frame_id);
                const int64_t bw_time_start = AscendC::GetSystemCycle();
                for (int g = 0; g < full_groups; ++g) {
                    for (int j = 0; j < batch_size; ++j) {
                        if (is_put) {
                            aclshmemx_sdma_qp_put_nbi(
                                dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements),
                                peer_pe, aiv_idx, EVENT_ID0);
                        } else {
                            aclshmemx_sdma_qp_get_nbi(
                                dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements),
                                peer_pe, aiv_idx, EVENT_ID0);
                        }
                    }
                    aclshmemx_sdma_qp_quiet(ub_ptr, ub_size, aiv_idx, EVENT_ID0);
                }
                for (int j = 0; j < remainder_iters; ++j) {
                    if (is_put) {
                        aclshmemx_sdma_qp_put_nbi(
                            dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements), peer_pe,
                            aiv_idx, EVENT_ID0);
                    } else {
                        aclshmemx_sdma_qp_get_nbi(
                            dst + offset, src + offset, ub_ptr, ub_size, static_cast<uint32_t>(local_elements), peer_pe,
                            aiv_idx, EVENT_ID0);
                    }
                }
                if (remainder_iters > 0) {
                    aclshmemx_sdma_qp_quiet(ub_ptr, ub_size, aiv_idx, EVENT_ID0);
                }
                const int64_t bw_time_end = AscendC::GetSystemCycle();
                SHMEMI_PROF_END(frame_id);
                if (timing_slot != nullptr) {
                    *timing_slot = bw_time_end - bw_time_start;
                }
            }
        }

        // Publish the per-AIV timing slot (flush to GM so host and remote SDMA reads see
        // it), then in bidirectional mode exchange it with the peer so both PEs end up
        // holding forward and reverse windows (rdma_perftest semantics). GET must pull
        // the peer slot because SDMA WRITE is unavailable on Ascend950.
        if (timing_slot != nullptr && has_traffic) {
            AscendC::PipeBarrier<PIPE_ALL>();
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(timing_slot));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        // The bidirectional GET exchange reads the peer's slot, so wait until every AIV on
        // both PEs has written and flushed its own slot. All AIVs (idle included) must
        // join or sync_vec_all deadlocks. The PUT exchange reads only the local slot and
        // needs no cross-PE sync.
        if (bidirectional && !is_put) {
            aclshmemx_sync_vec_all();
        }
        if (timing_slot != nullptr && has_traffic && bidirectional) {
            __gm__ T* slot_t = reinterpret_cast<__gm__ T*>(timing_slot);
            if (is_put) {
                // Push: own slot value -> the peer's same slot (the peer never writes it
                // locally, so both regions end up filled on both PEs).
                aclshmemx_sdma_qp_put_nbi(
                    slot_t, slot_t, ub_ptr, ub_size, sizeof(int64_t) / sizeof(T), peer_pe, aiv_idx, EVENT_ID0);
            } else {
                // Pull: the peer's slot value -> the local copy of that same slot. dst and
                // src keep the peer's region offset; using the own slot as dst would
                // overwrite the local timing with the peer's and leave one region at 0.
                const int peer_timing_region = (timing_region == 0) ? 1 : 0;
                __gm__ T* peer_slot_t =
                    reinterpret_cast<__gm__ T*>(timing_out + peer_timing_region * ACLSHMEM_MAX_AIV_PER_NPU + aiv_idx);
                aclshmemx_sdma_qp_get_nbi(
                    peer_slot_t, peer_slot_t, ub_ptr, ub_size, sizeof(int64_t) / sizeof(T), peer_pe, aiv_idx,
                    EVENT_ID0);
            }
            aclshmemx_sdma_qp_quiet(ub_ptr, ub_size, aiv_idx, EVENT_ID0);
        }

        // Collective sync after qp_quiet: every AIV on every PE joins, so all pending SDMA
        // data (including the timing exchange above) is guaranteed to be received before
        // the kernel exits.
        aclshmemx_sync_vec_all();
    }
}

#define DEFINE_SDMA_PERF_KERNEL(type_name, cpp_type)                                                                   \
    extern "C" __global__ __aicore__ void sdma_perf_##type_name(                                                       \
        uint64_t fftsAddr, GM_ADDR dst, GM_ADDR src, GM_ADDR timing_out, int elements, int32_t frame_id,               \
        int64_t prof_pe, int loop_count, int ub_size_kb, bool bidirectional, bool is_put, uint32_t qp_num, int metric, \
        int batch)                                                                                                     \
    {                                                                                                                  \
        sdma_perf_impl<cpp_type>(                                                                                      \
            fftsAddr, dst, src, timing_out, elements, frame_id, prof_pe, loop_count, ub_size_kb, bidirectional,        \
            is_put, qp_num, metric, batch);                                                                            \
    }

DEFINE_SDMA_PERF_KERNEL(float, float)
DEFINE_SDMA_PERF_KERNEL(int8, int8_t)
DEFINE_SDMA_PERF_KERNEL(int16, int16_t)
DEFINE_SDMA_PERF_KERNEL(int32, int32_t)
DEFINE_SDMA_PERF_KERNEL(int64, int64_t)
DEFINE_SDMA_PERF_KERNEL(uint8, uint8_t)
DEFINE_SDMA_PERF_KERNEL(uint16, uint16_t)
DEFINE_SDMA_PERF_KERNEL(uint32, uint32_t)
DEFINE_SDMA_PERF_KERNEL(uint64, uint64_t)
DEFINE_SDMA_PERF_KERNEL(char, char)

#define DISPATCH_SDMA_PERF(type_name, cpp_type)                                                          \
    sdma_perf_##type_name<<<block_dim, nullptr, stream>>>(                                               \
        fftsAddr, dst_gva, src_gva, timing_out_gva, elements, frame_id, prof_pe, loop_count, ub_size_kb, \
        bidirectional, is_put, qp_num, metric, batch)

extern "C" void launch_sdma_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t fftsAddr, uint8_t* dst_gva, uint8_t* src_gva, uint8_t* timing_out_gva,
    int elements, int32_t frame_id, int data_type, int64_t prof_pe, int loop_count, int ub_size_kb, bool bidirectional,
    bool is_put, uint32_t qp_num, int metric, int batch)
{
    perftest::perf_data_type_t type = static_cast<perftest::perf_data_type_t>(data_type);
    switch (type) {
        case perftest::DATA_TYPE_FLOAT:
            DISPATCH_SDMA_PERF(float, float);
            break;
        case perftest::DATA_TYPE_INT8:
            DISPATCH_SDMA_PERF(int8, int8_t);
            break;
        case perftest::DATA_TYPE_INT16:
            DISPATCH_SDMA_PERF(int16, int16_t);
            break;
        case perftest::DATA_TYPE_INT32:
            DISPATCH_SDMA_PERF(int32, int32_t);
            break;
        case perftest::DATA_TYPE_INT64:
            DISPATCH_SDMA_PERF(int64, int64_t);
            break;
        case perftest::DATA_TYPE_UINT8:
            DISPATCH_SDMA_PERF(uint8, uint8_t);
            break;
        case perftest::DATA_TYPE_UINT16:
            DISPATCH_SDMA_PERF(uint16, uint16_t);
            break;
        case perftest::DATA_TYPE_UINT32:
            DISPATCH_SDMA_PERF(uint32, uint32_t);
            break;
        case perftest::DATA_TYPE_UINT64:
            DISPATCH_SDMA_PERF(uint64, uint64_t);
            break;
        case perftest::DATA_TYPE_CHAR:
            DISPATCH_SDMA_PERF(char, char);
            break;
        default:
            DISPATCH_SDMA_PERF(float, float);
            break;
    }
}

#endif
