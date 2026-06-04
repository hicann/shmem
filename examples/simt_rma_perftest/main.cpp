/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "../utils/utils.h"
#include "utils/prof/shmemi_prof.h"

#include "argparser.h"

/*
 * SIMT RMA gm2gm Device-to-Device Performance Test
 *
 * Architecture:
 * - Requires 2 PEs: Active PE (PE0) and Passive PE (PE1). Only the Active PE
 *   issues transfers; put/get are one-sided RMA, so the Passive PE just acts
 *   as the remote endpoint (host-side barriers provide the synchronization).
 * - Each PE allocates (256KB * num_cores) of symmetric memory; 256KB is the
 *   per-core physical buffer (per_core_bytes), independent of the --ub_size arg.
 * - The symmetric memory is viewed as a logical ring: each core owns one
 *   contiguous logical segment of size logical_segment_bytes =
 *   min(256KB, (warmup + loops) * per_invocation_bytes), so the ring spans
 *   total_cores * logical_segment_bytes. Keeping the working set spread across
 *   distinct addresses limits data-cache reuse and avoids inflating bandwidth.
 *
 * Operations (src and dst are the SAME symmetric offset):
 * - Put: Active PE pushes data to Passive PE; Passive PE validates.
 * - Get: Active PE fetches data from Passive PE; Active PE validates.
 * - None: still calls the put API with count = 0, measuring call overhead only.
 *
 * Execution:
 * - Each iteration slides a window forward by per_invocation_bytes; every core
 *   transfers one chunk of that size, its base offset staggered by segment.
 * - Address offset wraps within the logical ring.
 * - Skips the first 'warmup' iterations, then averages over 'loops' iterations.
 */

// Change these values manually to reconfigure the test.
constexpr VfType VF_TYPE = VfType::Simt;   // Select from { VfType::Simt, VfType::Simd }
constexpr OpType OP_TYPE = OpType::Put;    // Select from { OpType::Put, OpType::Get, OpType::None }
constexpr int32_t DATA_SIZE = 32;             // Select from { 8, 16, 32, 64 }
constexpr int32_t THREAD_COUNT = 1024;

constexpr int32_t ACTIVE_PE  = 0;
constexpr int32_t PASSIVE_PE = 1;

const char *ipport = "tcp://127.0.0.1:8998";
aclshmemx_uniqueid_t default_flag_uid;
static aclshmem_prof_pe_t *out_profs;

// -----------------------------------------------------------------------------
// Data preparation and validation
// -----------------------------------------------------------------------------
class PerfDataPrepValidator 
{
private:
    int32_t my_pe;
    OpType OP_TYPE;
    int8_t expected_value;
public:
    PerfDataPrepValidator(int32_t pe, OpType type) : my_pe(pe), OP_TYPE(type) 
    {
        // Expected value depends on the source PE for the operation
        if (OP_TYPE == OpType::Put) {
            expected_value = ACTIVE_PE + 10;
        } else if (OP_TYPE == OpType::Get) {
            expected_value = PASSIVE_PE + 10;
        } else {
            expected_value = 0; // not used
        }
    }

    void init_data(void* data, size_t len) 
    {
        int8_t fill_val = my_pe + 10;
        std::fill_n(static_cast<int8_t*>(data), len, fill_val);
    }

    // Original full-range check (no longer used, kept for reference)
    bool check_data(void* data, size_t len)
    {
        // No validation on the source side
        if (
            (OP_TYPE == OpType::Put && my_pe == ACTIVE_PE) ||
            (OP_TYPE == OpType::Get && my_pe == PASSIVE_PE) ||
            OP_TYPE == OpType::None
        ) {
            return true;
        }
        const int8_t* result = static_cast<const int8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            if (result[i] != expected_value) {
                return false;
            }
        }
        return true;
    }

    // Region-based check: validates the first 'effective_bytes_per_block' of
    // each block. Called with block_stride == logical_segment_bytes so the
    // validated regions exactly match the contiguous ring written by the kernel.
    bool check_data_regions(void* data,
                            size_t effective_bytes_per_block,
                            int32_t block_count,
                            size_t block_stride)
    {
        // No validation on the source side
        if (
            (OP_TYPE == OpType::Put && my_pe == ACTIVE_PE) ||
            (OP_TYPE == OpType::Get && my_pe == PASSIVE_PE) ||
            OP_TYPE == OpType::None
        ) {
            return true;
        }
        const int8_t* base = static_cast<const int8_t*>(data);
        for (int32_t b = 0; b < block_count; ++b) {
            const int8_t* block_start = base + b * block_stride;
            for (size_t i = 0; i < effective_bytes_per_block; ++i) {
                if (block_start[i] != expected_value) {
                    return false;
                }
            }
        }
        return true;
    }
};

// -----------------------------------------------------------------------------
// Invoke parameters
// -----------------------------------------------------------------------------
struct InvokeParameters {
/*
 *  Each core has a 256KB physically allocated buffer (per_core_bytes).
 *  For the purpose of the global walk, we define a logical segment per core
 *  of size 'logical_segment_bytes', which equals min(per_core_bytes,
 *  (warmup+loops)*per_invocation_bytes). All cores collectively view the
 *  symmetric memory as a logical ring of size total_cores * logical_segment_bytes,
 *  partitioned into contiguous logical segments, one per core.
 *
 *  The outer loop (warmup+loops iterations) advances a sliding window across
 *  this ring by 'per_invocation_bytes' each step. The window consists of one
 *  contiguous chunk per core, each of size 'per_invocation_bytes', starting at
 *  offsets {window_start + core_id * logical_segment_bytes} (mod ring_size).
 *  Because per_invocation_bytes <= logical_segment_bytes, these chunks never
 *  overlap within a single iteration. SyncAll between iterations guarantees
 *  no cross-iteration conflicts.
 *
 *  Validation walks the buffer with the same layout: it checks each core's
 *  logical segment using 'logical_segment_bytes' as the stride (see the
 *  check_data_regions call), so the written ring and the validated regions
 *  coincide for any logical_segment_bytes (whether or not it equals 256KB).
 *  Note: when logical_segment_bytes < per_core_bytes the tail of each 256KB
 *  physical buffer is intentionally left untouched and not validated.
 */
public:
    constexpr static int32_t per_core_bytes = 256 * 1024;    // physical buffer size per core
    int32_t per_invocation_bytes;                            // transfer size per iteration
    int32_t logical_segment_bytes;                           // logical region size per core
    int32_t warmup;
    int32_t loops;

    InvokeParameters(int32_t invoke_exp, int32_t warmup, int32_t loops) : warmup(warmup), loops(loops)
    {
        per_invocation_bytes = 1 << invoke_exp;

        int64_t total_requested = (warmup + loops) * static_cast<int64_t>(per_invocation_bytes);
        logical_segment_bytes = total_requested;
        if (total_requested > per_core_bytes) {
            logical_segment_bytes = per_core_bytes;
        }
        // Ensure at least one transfer fits into the segment
        if (total_requested < per_invocation_bytes) {
            logical_segment_bytes = per_invocation_bytes;
        }
    }
};

// -----------------------------------------------------------------------------
// Transfer kernels (single definition per compilation unit)
// -----------------------------------------------------------------------------
__simt_vf__ __launch_bounds__(THREAD_COUNT) inline void transfer_vf_kernel(
    __gm__ void* dst, __gm__ void* src, size_t count, int32_t pe)
{
    if constexpr (OP_TYPE == OpType::None) {
        if constexpr (DATA_SIZE == 8)       simt::aclshmemx_put8_block(dst, src, 0, pe);
        else if constexpr (DATA_SIZE == 16) simt::aclshmemx_put16_block(dst, src, 0, pe);
        else if constexpr (DATA_SIZE == 32) simt::aclshmemx_put32_block(dst, src, 0, pe);
        else if constexpr (DATA_SIZE == 64) simt::aclshmemx_put64_block(dst, src, 0, pe);
        else static_assert(DATA_SIZE == 8 || DATA_SIZE == 16 || DATA_SIZE == 32 || DATA_SIZE == 64,
                           "Unsupported DATA_SIZE");
    } else {
        if constexpr (OP_TYPE == OpType::Put) {
            if constexpr (DATA_SIZE == 8)       simt::aclshmemx_put8_block(dst, src, count, pe);
            else if constexpr (DATA_SIZE == 16) simt::aclshmemx_put16_block(dst, src, count, pe);
            else if constexpr (DATA_SIZE == 32) simt::aclshmemx_put32_block(dst, src, count, pe);
            else if constexpr (DATA_SIZE == 64) simt::aclshmemx_put64_block(dst, src, count, pe);
            else static_assert(DATA_SIZE == 8 || DATA_SIZE == 16 || DATA_SIZE == 32 || DATA_SIZE == 64,
                               "Unsupported DATA_SIZE");
        } else { // OpType::Get
            if constexpr (DATA_SIZE == 8)       simt::aclshmemx_get8_block(dst, src, count, pe);
            else if constexpr (DATA_SIZE == 16) simt::aclshmemx_get16_block(dst, src, count, pe);
            else if constexpr (DATA_SIZE == 32) simt::aclshmemx_get32_block(dst, src, count, pe);
            else if constexpr (DATA_SIZE == 64) simt::aclshmemx_get64_block(dst, src, count, pe);
            else static_assert(DATA_SIZE == 8 || DATA_SIZE == 16 || DATA_SIZE == 32 || DATA_SIZE == 64,
                               "Unsupported DATA_SIZE");
        }
    }
}

__aicore__ inline void transfer_kernel(__gm__ void* dst, __gm__ void* src,
                                       size_t count, int32_t pe, int32_t ub_size)
{
    if constexpr (VF_TYPE == VfType::Simd) {
        using T = typename TraitTypeFromDataSize<DATA_SIZE>::type;
        if constexpr (OP_TYPE == OpType::None) {
            aclshmemx_mte_put_nbi((__gm__ T*)dst, (__gm__ T*)src,
                                  (__ubuf__ T*)(0), ub_size, 0, pe, 0);
        } else if constexpr (OP_TYPE == OpType::Put) {
            aclshmemx_mte_put_nbi((__gm__ T*)dst, (__gm__ T*)src,
                                  (__ubuf__ T*)(0), ub_size, count, pe, 0);
        } else { // OpType::Get
            aclshmemx_mte_get_nbi((__gm__ T*)dst, (__gm__ T*)src,
                                  (__ubuf__ T*)(0), ub_size, count, pe, 0);
        }
        aclshmemx_mte_quiet();

    } else if constexpr (VF_TYPE == VfType::Simt) {
        asc_vf_call<transfer_vf_kernel>(dim3(THREAD_COUNT), dst, src, count, pe);
    } else {
        static_assert(VF_TYPE == VfType::Simt || VF_TYPE == VfType::Simd,
                      "Unsupported VF_TYPE");
    }
}

// -----------------------------------------------------------------------------
// Vector / Scalar synchronization
// -----------------------------------------------------------------------------
template <VfType vf>
struct VSSync { };

template <>
struct VSSync<VfType::Simt> {
    __aicore__ static inline void start() 
    {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    }

    __aicore__ static inline void end() 
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    }
};

template <>
struct VSSync<VfType::Simd> {
    __aicore__ static inline void start() 
    {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    }

    __aicore__ static inline void end() 
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }
};

// -----------------------------------------------------------------------------
// Main test kernel
// -----------------------------------------------------------------------------
__global__ __vector__ void run_demo_mem(
    __gm__ void* sym_addr,
    InvokeParameters invp,
    int32_t ub_size,
    int32_t frame_id
) 
{
    int32_t my_pe = aclshmem_my_pe();
    int32_t n_pes = aclshmem_n_pes();
    int32_t next_pe = (my_pe + 1) % n_pes;

    int32_t my_block_id = AscendC::GetBlockIdx();
    int32_t total_block_num = AscendC::GetBlockNum();

    if (my_pe != ACTIVE_PE) {
        return;
    }

    // Total size of the logical ring = total_blocks * logical_segment_bytes
    const int32_t ring_size = total_block_num * invp.logical_segment_bytes;
    // Number of window steps to scan the whole ring once. logical_segment_bytes
    // is always a multiple of per_invocation_bytes, so this division is exact
    // and window_start sweeps the ring with no gap and no overrun.
    const int32_t steps_per_ring_scan = ring_size / invp.per_invocation_bytes;

    // Outer loop: each iteration moves the global window by per_invocation_bytes.
    for (int32_t i = 0; i < invp.warmup + invp.loops; i++) {
        // Global window start offset (in the logical ring) for this iteration.
        int32_t window_start = (i % steps_per_ring_scan) * invp.per_invocation_bytes;

        // This block's logical segment base offset = block_id * logical_segment_bytes.
        // Add the window start and wrap around the ring.
        int32_t offset = (window_start + my_block_id * invp.logical_segment_bytes) % ring_size;

        __gm__ int8_t* p_temp = (__gm__ int8_t*)sym_addr;
        p_temp += offset;
        __gm__ void* input_ptr = (__gm__ void*)p_temp;

        size_t count = invp.per_invocation_bytes / (DATA_SIZE / 8);
        
        if (i >= invp.warmup) {
            SHMEMI_PROF_START(frame_id);
        }
        
        VSSync<VF_TYPE>::start();

        transfer_kernel(input_ptr, input_ptr, count, next_pe, ub_size);

        AscendC::SyncAll<true>();
        VSSync<VF_TYPE>::end();

        if (i >= invp.warmup) {
            SHMEMI_PROF_END(frame_id);
        }
    }
}

// -----------------------------------------------------------------------------
// Host-side execution
// -----------------------------------------------------------------------------
int32_t execute_test_loop(const Config& config, aclrtStream stream) 
{
    int npes = config.npes;
    int mype = config.mype;
    std::vector<std::vector<std::string>> csv_data = {
        {"DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s", "CoreMaxTime/us", "SingleCoreTime/us"},
    };

    constexpr size_t per_core_bytes = InvokeParameters::per_core_bytes;
    size_t max_bytes = per_core_bytes * config.used_core;
    
    void* host_mem = nullptr;
    void* device_mem = nullptr;
    PerfDataPrepValidator pdpv(mype, OP_TYPE);

    ACL_CHECK_WITH_RET(aclrtMallocHost(&host_mem, max_bytes), ERROR_LOG("Failed to allocate host memory"), return -1);
    device_mem = aclshmem_malloc(max_bytes);
    // If device allocation fails, release the host memory acquired above before returning.
    ACL_CHECK_WITH_RET(device_mem == nullptr, ERROR_LOG("Failed to allocate device memory"),
                       { aclrtFreeHost(host_mem); return -1; });

    bool ever_failed = false;
    for (int32_t i = 0; i <= config.bytes_in_exp_max - config.bytes_in_exp_min; i++) {
        int32_t exp = config.bytes_in_exp_min + i;
        int32_t frame_id = i;
        
        // Initialize entire buffer with PE-specific pattern
        pdpv.init_data(host_mem, max_bytes);
        InvokeParameters invp(exp, config.warmup_loops, config.loops);
        
        ACL_CHECK(aclrtMemcpy(device_mem, max_bytes, host_mem, max_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

        aclshmem_barrier_all();
        run_demo_mem<<<config.used_core, 0, stream>>>(
            device_mem,
            invp,
            config.ub_size * 1024,
            frame_id
        );

        ACL_CHECK(aclrtSynchronizeStream(stream));
        aclshmem_barrier_all();

        ACL_CHECK(aclrtMemcpy(host_mem, max_bytes, device_mem, max_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

        // Validate only the effectively used regions within each block
        bool success = pdpv.check_data_regions(host_mem,
                                       invp.logical_segment_bytes,
                                       config.used_core,
                                       invp.logical_segment_bytes);
        ever_failed = ever_failed || (!success);

        aclshmemx_show_prof(&out_profs, false);
        collect_prof_data_to_csv(
            out_profs,
            frame_id,
            config.used_core,
            invp.per_invocation_bytes,
            config.npes, config.ub_size,
            csv_data
        );
    }

    aclshmem_free(device_mem);
    aclrtFreeHost(host_mem);

    aclshmemx_show_prof(nullptr, true);

    if (config.mype == ACTIVE_PE) {
        std::string csv_filename = 
            "output/" + 
            std::to_string(DATA_SIZE) + "_" +
            std::to_string(config.used_core) + "_" +
            to_string(OP_TYPE) + "_" + 
            to_string(VF_TYPE) + "_" + 
            std::to_string(config.bytes_in_exp_min) + "-" + std::to_string(config.bytes_in_exp_max) + "_" +
            "w" + std::to_string(config.warmup_loops) + "l" + std::to_string(config.loops) + "_" +
            "t" + std::to_string(THREAD_COUNT) + "_" + 
            ".csv";
        write_csv(csv_filename, csv_data);
    }

    if (ever_failed) {
        printf("[ERROR] Some tests failed.\n");
    }
    return ever_failed ? -1 : 0;
}

int32_t run_config_test(const Config& config) {
    printf("[INFO] Starting test for PE %d of %d\n", config.mype, config.npes);

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(config.mype));
    ACL_CHECK(aclrtCreateStream(&stream));
    
    printf("[INFO] Initializing ACLSHMEM...\n");
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(config.mype, config.npes, local_mem_size, ipport, default_flag_uid, &attributes);
    // ACLSHMEM is not yet initialized here, so on failure we must NOT call
    // aclshmem_finalize; unwind only what was already acquired above
    // (stream, device, ACL init) in reverse order.
    ACL_CHECK_WITH_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
                       ERROR_LOG("aclshmemx_init failed"),
                       {
                           aclrtDestroyStream(stream);
                           aclrtResetDevice(config.mype);
                           aclFinalize();
                           return -1;
                       });

    int32_t ret = execute_test_loop(config, stream);
    
    printf("[INFO] Cleaning up resources...\n");
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(config.mype);
    aclFinalize();

    return ret;
}

int32_t main(int argc, char* argv[]) {
    setenv("SHMEM_CYCLE_PROF_PE", "0", 1);

    // Handle --help up front so it exits successfully (0) instead of being
    // conflated with a parse error below.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    auto result = parse_args(argc, argv);

    if (!result) {
        // parse_args has already written the specific reason to stderr;
        // print the usage so the user can see how to correct the invocation.
        std::cerr << "[ERROR] Invalid arguments.\n";
        print_usage(argv[0]);
        return 1;
    }
    Config config = *result;
    int32_t ret = run_config_test(config);

    return ret;
}