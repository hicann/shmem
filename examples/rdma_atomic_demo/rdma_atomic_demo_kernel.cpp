/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"
#include "rdma_atomic_demo_kernel.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void rdma_atomic_demo_kernel(
    GM_ADDR target, GM_ADDR result)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
    // The first UB allocation matches the workspace configured on the Host at offset 0.
    pipe.InitBuffer(buffer, UB_BYTES);

    AscendC::PipeBarrier<PIPE_ALL>();
    // All targets must be initialized before any PE starts its atomic operations.
    aclshmemx_roce_barrier_all();

    const int peer = (aclshmem_my_pe() + 1) % aclshmem_n_pes();
    const uint64_t initial = INITIAL_VALUE + peer;
    auto dst = reinterpret_cast<__gm__ uint64_t*>(target);

    const uint64_t faa_old = aclshmemx_roce_atomic_fetch_add(dst, ADD_VALUE, peer);
    const uint64_t cas_old =
        aclshmemx_roce_atomic_compare_swap(dst, initial + ADD_VALUE, initial + 2 * ADD_VALUE, peer);
    // The previous CAS changed the value, so this comparison must fail and leave the target unchanged.
    const uint64_t cas_miss_old =
        aclshmemx_roce_atomic_compare_swap(dst, initial + ADD_VALUE, initial + 3 * ADD_VALUE, peer);

    // Fetching atomics complete before returning; their UB workspace can now hold the return values.
    AscendC::LocalTensor<uint64_t> old_values = buffer.Get<uint64_t>();
    old_values.SetValue(0, faa_old);
    old_values.SetValue(1, cas_old);
    old_values.SetValue(2, cas_miss_old);
    old_values.SetValue(3, 0);
    AscendC::GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(result));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::DataCopy(output, old_values, 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);

    // Complete incoming atomics before the Host reads or frees the local target.
    aclshmemx_roce_barrier_all();
}

void launch_rdma_atomic_demo(uint32_t block_dim, void* stream, uint8_t* target, uint8_t* result)
{
    rdma_atomic_demo_kernel<<<block_dim, nullptr, stream>>>(target, result);
}
