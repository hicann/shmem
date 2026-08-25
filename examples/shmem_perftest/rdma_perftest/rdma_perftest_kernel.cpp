/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RDMAPERF_KERNEL_RDMA_PERFTEST_
#define _RDMAPERF_KERNEL_RDMA_PERFTEST_

#include "kernel_operator.h"
#include "shmem.h"
#include "perftest_common_types.h"

constexpr uint64_t XSCALE_MULTI_QP_AGGREGATE_THRESHOLD = 64 * 1024;

template <typename T>
__aicore__ inline void rdma_perf_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_num, int qp_index,
    uint32_t op_idx, uint32_t sync_id)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    const uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool multi_core = AscendC::GetBlockNum() > 1;
    const uint32_t selected_qp =
        qp_index >= 0 ? static_cast<uint32_t>(qp_index) : (multi_core ? block_id : op_idx % qp_num);
    if constexpr (use_qp_specific) {
        aclshmemx_roce_qp_put_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id);
    } else {
        aclshmemx_roce_put_nbi(dst, src, buf, elem_size, pe, sync_id);
    }
}

template <typename T>
__aicore__ inline void rdma_perf_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t qp_num, int qp_index,
    uint32_t op_idx, uint32_t sync_id)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    const uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool multi_core = AscendC::GetBlockNum() > 1;
    const uint32_t selected_qp =
        qp_index >= 0 ? static_cast<uint32_t>(qp_index) : (multi_core ? block_id : op_idx % qp_num);
    if constexpr (use_qp_specific) {
        aclshmemx_roce_qp_get_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id);
    } else {
        aclshmemx_roce_get_nbi(dst, src, buf, elem_size, pe, sync_id);
    }
}

#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
template <typename T>
__aicore__ inline void rdma_perf_put_aggregate_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t selected_qp, uint32_t op_count,
    uint32_t sync_id)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);
    for (uint32_t op = 0; op < op_count; ++op) {
        const bool is_last = op + 1 == op_count;
        if (is_last) {
            aclshmemx_roce_qp_put_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, submit_action);
        } else {
            aclshmemx_roce_qp_put_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, defer_action);
        }
    }
}

template <typename T>
__aicore__ inline void rdma_perf_get_aggregate_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t selected_qp, uint32_t op_count,
    uint32_t sync_id)
{
    aclshmemx_submit_state_t submit_state{};
    aclshmemx_defer_t defer_action(submit_state);
    aclshmemx_submit_t submit_action(submit_state);
    for (uint32_t op = 0; op < op_count; ++op) {
        const bool is_last = op + 1 == op_count;
        if (is_last) {
            aclshmemx_roce_qp_get_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, submit_action);
        } else {
            aclshmemx_roce_qp_get_nbi(dst, src, buf, elem_size, pe, selected_qp, sync_id, defer_action);
        }
    }
}
#endif

__aicore__ inline bool rdma_perf_should_aggregate(uint32_t qp_num, uint64_t msg_size, int metric)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    return metric == static_cast<int>(perftest::PERF_METRIC_BW) && qp_num <= 2 &&
           msg_size < XSCALE_MULTI_QP_AGGREGATE_THRESHOLD;
#else
    (void)qp_num;
    (void)msg_size;
    (void)metric;
    return false;
#endif
}

template <typename T>
__aicore__ inline void rdma_perf_quiet(
    uint32_t pe, __ubuf__ T* buf, uint32_t qp_num, uint32_t op_count, int qp_index, uint32_t qp_start_idx,
    uint32_t sync_id)
{
    constexpr bool use_qp_specific = ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::XSCALE ||
                                     ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825;
    const uint32_t block_id = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const bool multi_core = AscendC::GetBlockNum() > 1;
    if constexpr (use_qp_specific) {
        if (qp_index >= 0) {
            aclshmemx_roce_qp_quiet(pe, static_cast<uint32_t>(qp_index), buf, sync_id);
            return;
        }
        if (multi_core) {
            aclshmemx_roce_qp_quiet(pe, block_id, buf, sync_id);
            return;
        }
        if (qp_num == 1) {
            aclshmemx_roce_qp_quiet(pe, 0, buf, sync_id);
            return;
        }

        uint32_t quiet_qp_count = op_count < qp_num ? op_count : qp_num;
        for (uint32_t qp_idx = 0; qp_idx < quiet_qp_count; ++qp_idx) {
            aclshmemx_roce_qp_quiet(pe, (qp_start_idx + qp_idx) % qp_num, buf, sync_id);
        }
    } else {
        aclshmemx_roce_quiet(pe, buf, sync_id);
    }
}

template <typename T>
__aicore__ inline void rdma_perf_test_put_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, int ub_size_b,
    int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index, GM_ADDR timing_out_gva)
{
    util_set_ffts_config(fftsAddr);
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();
    __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);

    bool is_bidir = (test_mode == perftest::TEST_MODE_RDMA_BI_PUT);
    const bool use_multi_core =
        (metric == static_cast<int>(perftest::PERF_METRIC_BW) && qp_num > 1 && qp_index < 0 &&
         AscendC::GetBlockNum() > 1);
    const uint32_t block_id = use_multi_core ? static_cast<uint32_t>(AscendC::GetBlockIdx()) : 0;
    const bool all_blocks_active = is_bidir || pe == 0;

    if (use_multi_core && !is_bidir && pe != 0 && block_id != 0) {
        return;
    }
    if (use_multi_core) {
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        // Ensure both PEs are synchronized before any RDMA operations begin
        aclshmemx_roce_barrier_all();
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
    }

    if (!is_bidir && pe != 0) {
        if (timing_out != nullptr) {
            timing_out[1] = 0;
        }
        aclshmemx_roce_barrier_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_b);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_b, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ubLocal.GetPhyAddr());

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);
    if (use_multi_core) {
        const uint64_t qp_region_offset = static_cast<uint64_t>(elements) * block_id;
        dst_gm += qp_region_offset;
        src_gm += qp_region_offset;
    }

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (loop_test <= 0 || batch <= 0 || batch > loop_test) ? loop_test : batch;
    uint32_t qp_op_seq = 0;

    const uint64_t msg_size = static_cast<uint64_t>(elements) * sizeof(T);
    const bool use_aggregate = rdma_perf_should_aggregate(qp_num, msg_size, metric);
    const uint32_t aggregate_qp = qp_index >= 0 ? static_cast<uint32_t>(qp_index) : block_id;

    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // Latency: time loop_count nbi submits in a single window, quiet outside.
        const uint32_t warmup_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            rdma_perf_put_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(warmup), sync_id);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                rdma_perf_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, warmup, qp_index, warmup_qp_start % qp_num, sync_id);
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_start = AscendC::GetSystemCycle();
        const uint32_t loop_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && loop_test > 1) {
            rdma_perf_put_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(loop_test), sync_id);
        } else
#endif
        {
            for (int i = 0; i < loop_test; ++i) {
                rdma_perf_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_end = AscendC::GetSystemCycle();
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, loop_test, qp_index, loop_qp_start % qp_num, sync_id);
        if (timing_out != nullptr) {
            int64_t api_total_time = api_time_end - api_time_start;
            if (pe == 0) {
                timing_out[0] = api_total_time;
            } else {
                timing_out[1] = api_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                const uint32_t timing_qp_start = qp_op_seq;
                rdma_perf_put_nbi(
                    slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, qp_num, qp_index, qp_op_seq++, sync_id);
                rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, timing_qp_start % qp_num, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }

    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
        const uint32_t warmup_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            rdma_perf_put_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(warmup), sync_id);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                rdma_perf_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, warmup, qp_index, warmup_qp_start % qp_num, sync_id);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t waiting_time_start = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                waiting_time_start = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            waiting_time_start = AscendC::GetSystemCycle();
        }
        for (int g = 0; g < full_groups; ++g) {
            const uint32_t group_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
            if (use_aggregate && batch_size > 1) {
                rdma_perf_put_aggregate_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                    static_cast<uint32_t>(batch_size), sync_id);
            } else
#endif
            {
                for (int j = 0; j < batch_size; ++j) {
                    rdma_perf_put_nbi(
                        dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                        sync_id);
                }
            }
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, batch_size, qp_index, group_qp_start % qp_num, sync_id);
        }
        const uint32_t remainder_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && remainder > 1) {
            rdma_perf_put_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(remainder), sync_id);
        } else
#endif
        {
            for (int j = 0; j < remainder; ++j) {
                rdma_perf_put_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        if (remainder > 0) {
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, remainder, qp_index, remainder_qp_start % qp_num, sync_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t waiting_time_end = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                waiting_time_end = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            waiting_time_end = AscendC::GetSystemCycle();
        }

        // Output: PE0 -> timing_out[0], PE1 -> timing_out[1]
        if (timing_out != nullptr && (!use_multi_core || block_id == 0)) {
            int64_t waiting_total_time = waiting_time_end - waiting_time_start;
            if (pe == 0) {
                timing_out[0] = waiting_total_time;
            } else {
                timing_out[1] = waiting_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                const uint32_t timing_qp_start = qp_op_seq;
                rdma_perf_put_nbi(
                    slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, qp_num, qp_index, qp_op_seq++, sync_id);
                rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, timing_qp_start % qp_num, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        aclshmemx_roce_barrier_all();
    }
}

template <typename T>
__aicore__ inline void rdma_perf_test_get_impl(
    uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode, int ub_size_b,
    int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index, GM_ADDR timing_out_gva)
{
    util_set_ffts_config(fftsAddr);
    int64_t pe = aclshmem_my_pe();
    int peer_pe = (pe + 1) % aclshmem_n_pes();
    __gm__ int64_t* timing_out = reinterpret_cast<__gm__ int64_t*>(timing_out_gva);

    bool is_bidir = (test_mode == perftest::TEST_MODE_RDMA_BI_GET);
    const bool use_multi_core =
        (metric == static_cast<int>(perftest::PERF_METRIC_BW) && qp_num > 1 && qp_index < 0 &&
         AscendC::GetBlockNum() > 1);
    const uint32_t block_id = use_multi_core ? static_cast<uint32_t>(AscendC::GetBlockIdx()) : 0;
    const bool all_blocks_active = is_bidir || pe == 0;

    if (use_multi_core && !is_bidir && pe != 0 && block_id != 0) {
        return;
    }
    if (use_multi_core) {
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        // Ensure both PEs are synchronized before any RDMA operations begin
        aclshmemx_roce_barrier_all();
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
    }

    if (!is_bidir && pe != 0) {
        if (timing_out != nullptr) {
            timing_out[1] = 0;
        }
        aclshmemx_roce_barrier_all();
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, ub_size_b);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(ub_size_b, 0);
    __ubuf__ T* ub_ptr = reinterpret_cast<__ubuf__ T*>(ubLocal.GetPhyAddr());

    __gm__ T* dst_gm = reinterpret_cast<__gm__ T*>(dst_gva);
    __gm__ T* src_gm = reinterpret_cast<__gm__ T*>(src_gva);
    if (use_multi_core) {
        const uint64_t qp_region_offset = static_cast<uint64_t>(elements) * block_id;
        dst_gm += qp_region_offset;
        src_gm += qp_region_offset;
    }

    int warmup = perftest::PERFTEST_WARMUP_ITERS;
    int loop_test = loop_count;
    int batch_size = (loop_test <= 0 || batch <= 0 || batch > loop_test) ? loop_test : batch;
    uint32_t qp_op_seq = 0;

    const uint64_t msg_size = static_cast<uint64_t>(elements) * sizeof(T);
    const bool use_aggregate = rdma_perf_should_aggregate(qp_num, msg_size, metric);
    const uint32_t aggregate_qp = qp_index >= 0 ? static_cast<uint32_t>(qp_index) : block_id;

    AscendC::PipeBarrier<PIPE_ALL>();

    if (metric == static_cast<int>(perftest::PERF_METRIC_LAT)) {
        // Latency: time loop_count get_nbi submits in a single window, quiet outside.
        const uint32_t warmup_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            rdma_perf_get_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(warmup), sync_id);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                rdma_perf_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, warmup, qp_index, warmup_qp_start % qp_num, sync_id);
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_start = AscendC::GetSystemCycle();
        const uint32_t loop_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && loop_test > 1) {
            rdma_perf_get_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(loop_test), sync_id);
        } else
#endif
        {
            for (int i = 0; i < loop_test; ++i) {
                rdma_perf_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t api_time_end = AscendC::GetSystemCycle();
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, loop_test, qp_index, loop_qp_start % qp_num, sync_id);
        // dcci removed — relying on host-side 64B-aligned allocation instead
        if (timing_out != nullptr) {
            int64_t api_total_time = api_time_end - api_time_start;
            if (pe == 0) {
                timing_out[0] = api_total_time;
            } else {
                timing_out[1] = api_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                const uint32_t timing_qp_start = qp_op_seq;
                rdma_perf_put_nbi(
                    slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, qp_num, qp_index, qp_op_seq++, sync_id);
                rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, timing_qp_start % qp_num, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }

    } else {
        // Bandwidth: submit NBIs in groups of `batch_size`, quiet at each group boundary.
        const uint32_t warmup_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && warmup > 1) {
            rdma_perf_get_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(warmup), sync_id);
        } else
#endif
        {
            for (int i = 0; i < warmup; ++i) {
                rdma_perf_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        rdma_perf_quiet(peer_pe, ub_ptr, qp_num, warmup, qp_index, warmup_qp_start % qp_num, sync_id);
        int full_groups = loop_test / batch_size;
        int remainder = loop_test - full_groups * batch_size;
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t get_time_start = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                get_time_start = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            get_time_start = AscendC::GetSystemCycle();
        }
        for (int g = 0; g < full_groups; ++g) {
            const uint32_t group_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
            if (use_aggregate && batch_size > 1) {
                rdma_perf_get_aggregate_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                    static_cast<uint32_t>(batch_size), sync_id);
            } else
#endif
            {
                for (int j = 0; j < batch_size; ++j) {
                    rdma_perf_get_nbi(
                        dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                        sync_id);
                }
            }
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, batch_size, qp_index, group_qp_start % qp_num, sync_id);
        }
        const uint32_t remainder_qp_start = qp_op_seq;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
        if (use_aggregate && remainder > 1) {
            rdma_perf_get_aggregate_nbi(
                dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, aggregate_qp,
                static_cast<uint32_t>(remainder), sync_id);
        } else
#endif
        {
            for (int j = 0; j < remainder; ++j) {
                rdma_perf_get_nbi(
                    dst_gm, src_gm, ub_ptr, static_cast<uint32_t>(elements), peer_pe, qp_num, qp_index, qp_op_seq++,
                    sync_id);
            }
        }
        if (remainder > 0) {
            rdma_perf_quiet(peer_pe, ub_ptr, qp_num, remainder, qp_index, remainder_qp_start % qp_num, sync_id);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t get_time_end = 0;
        if (use_multi_core && all_blocks_active) {
            AscendC::SyncAll<true>();
            if (block_id == 0) {
                get_time_end = AscendC::GetSystemCycle();
            }
            AscendC::SyncAll<true>();
        } else {
            get_time_end = AscendC::GetSystemCycle();
        }

        // dcci removed — relying on host-side 64B-aligned allocation instead
        // Output: PE0 -> timing_out[0], PE1 -> timing_out[1]
        if (timing_out != nullptr && (!use_multi_core || block_id == 0)) {
            int64_t get_total_time = get_time_end - get_time_start;
            if (pe == 0) {
                timing_out[0] = get_total_time;
            } else {
                timing_out[1] = get_total_time;
            }
            dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(timing_out), sizeof(uint64_t) * 2);
            if (is_bidir) {
                aclshmemx_roce_barrier_all();
                __gm__ T* slot = reinterpret_cast<__gm__ T*>(&timing_out[pe]);
                const uint32_t timing_qp_start = qp_op_seq;
                rdma_perf_put_nbi(
                    slot, slot, ub_ptr, sizeof(int64_t) / sizeof(T), peer_pe, qp_num, qp_index, qp_op_seq++, sync_id);
                rdma_perf_quiet(peer_pe, ub_ptr, qp_num, 1, qp_index, timing_qp_start % qp_num, sync_id);
                aclshmemx_roce_barrier_all();
            }
        }
    }
    if (use_multi_core && all_blocks_active) {
        AscendC::SyncAll<true>();
        if (block_id == 0) {
            aclshmemx_roce_barrier_all();
        }
    } else {
        aclshmemx_roce_barrier_all();
    }
}

#define DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(type_name, cpp_type)                                                       \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_perf_test_##type_name##_put(           \
        uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode,         \
        int ub_size_b, int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index,      \
        GM_ADDR timing_out_gva)                                                                                     \
    {                                                                                                               \
        rdma_perf_test_put_impl<cpp_type>(                                                                          \
            fftsAddr, dst_gva, src_gva, elements, test_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, \
            qp_index, timing_out_gva);                                                                              \
    }                                                                                                               \
    extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_perf_test_##type_name##_get(           \
        uint64_t fftsAddr, GM_ADDR dst_gva, GM_ADDR src_gva, int elements, perftest::rdma_mode_t test_mode,         \
        int ub_size_b, int loop_count, int metric, int batch, uint32_t sync_id, uint32_t qp_num, int qp_index,      \
        GM_ADDR timing_out_gva)                                                                                     \
    {                                                                                                               \
        rdma_perf_test_get_impl<cpp_type>(                                                                          \
            fftsAddr, dst_gva, src_gva, elements, test_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, \
            qp_index, timing_out_gva);                                                                              \
    }

DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(float, float)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int8, int8_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int16, int16_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int32, int32_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(int64, int64_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint8, uint8_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint16, uint16_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint32, uint32_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(uint64, uint64_t)
DEFINE_RDMA_PERF_KERNEL_FOR_TYPE(char, char)

#define DISPATCH_RDMA_PERF_PUT(type_name, cpp_type)                                                                    \
    rdma_perf_test_##type_name##_put<<<block_dim, nullptr, stream>>>(                                                  \
        fftsAddr, dst_gva, src_gva, elements, t_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, qp_index, \
        timing_out_gva)

#define DISPATCH_RDMA_PERF_GET(type_name, cpp_type)                                                                    \
    rdma_perf_test_##type_name##_get<<<block_dim, nullptr, stream>>>(                                                  \
        fftsAddr, dst_gva, src_gva, elements, t_mode, ub_size_b, loop_count, metric, batch, sync_id, qp_num, qp_index, \
        timing_out_gva)

#define DISPATCH_RDMA_PERF_FOR_ALL_TYPES(MACRO) \
    switch (d_type) {                           \
        case perftest::DATA_TYPE_FLOAT:         \
            MACRO(float, float);                \
            break;                              \
        case perftest::DATA_TYPE_INT8:          \
            MACRO(int8, int8_t);                \
            break;                              \
        case perftest::DATA_TYPE_INT16:         \
            MACRO(int16, int16_t);              \
            break;                              \
        case perftest::DATA_TYPE_INT32:         \
            MACRO(int32, int32_t);              \
            break;                              \
        case perftest::DATA_TYPE_INT64:         \
            MACRO(int64, int64_t);              \
            break;                              \
        case perftest::DATA_TYPE_UINT8:         \
            MACRO(uint8, uint8_t);              \
            break;                              \
        case perftest::DATA_TYPE_UINT16:        \
            MACRO(uint16, uint16_t);            \
            break;                              \
        case perftest::DATA_TYPE_UINT32:        \
            MACRO(uint32, uint32_t);            \
            break;                              \
        case perftest::DATA_TYPE_UINT64:        \
            MACRO(uint64, uint64_t);            \
            break;                              \
        case perftest::DATA_TYPE_CHAR:          \
            MACRO(char, char);                  \
            break;                              \
        default:                                \
            MACRO(float, float);                \
            break;                              \
    }

extern "C" void launch_rdma_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t fftsAddr, uint8_t* dst_gva, uint8_t* src_gva, int elements,
    int test_mode, int data_type, int ub_size_b, int loop_count, int metric, int batch, int sync_id, int qp_num,
    int qp_index, uint8_t* timing_out_gva)
{
    perftest::rdma_mode_t t_mode = static_cast<perftest::rdma_mode_t>(test_mode);
    perftest::perf_data_type_t d_type = static_cast<perftest::perf_data_type_t>(data_type);

    switch (t_mode) {
        case perftest::TEST_MODE_RDMA_PUT:
        case perftest::TEST_MODE_RDMA_BI_PUT:
            DISPATCH_RDMA_PERF_FOR_ALL_TYPES(DISPATCH_RDMA_PERF_PUT);
            break;
        case perftest::TEST_MODE_RDMA_GET:
        case perftest::TEST_MODE_RDMA_BI_GET:
            DISPATCH_RDMA_PERF_FOR_ALL_TYPES(DISPATCH_RDMA_PERF_GET);
            break;
        default:
            break;
    }
}

#endif // _RDMAPERF_KERNEL_RDMA_PERFTEST_
