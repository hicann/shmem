/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_buffer_kernel.h"

#include "kernel_operator.h"
#include "shmem.h"

namespace {

constexpr uint32_t kMteUbBytes = 4096;
constexpr uint32_t kUdmaSyncId = 0;

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UserBufferMtePutKernel(
    uint64_t fftsConfig, GM_ADDR dst, GM_ADDR src, uint32_t bytes, int32_t peer)
{
    util_set_ffts_config(fftsConfig);
    auto* dstGlobal = reinterpret_cast<__gm__ uint8_t*>(dst);
    auto* srcGlobal = reinterpret_cast<__gm__ uint8_t*>(src);
    auto* copyUb = reinterpret_cast<__ubuf__ uint8_t*>(0);

    aclshmemx_mte_put_nbi<uint8_t>(dstGlobal, srcGlobal, copyUb, kMteUbBytes, bytes, peer, EVENT_ID0);
    aclshmemx_mte_quiet();
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void UserBufferUdmaPutKernel(
    GM_ADDR dst, GM_ADDR src, uint32_t bytes, int32_t peer)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> wqeBuffer;
    pipe.InitBuffer(wqeBuffer, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
    auto wqeTensor = wqeBuffer.GetWithOffset<uint8_t>(ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE, 0);
    auto* wqeScratch = reinterpret_cast<__ubuf__ uint8_t*>(wqeTensor.GetPhyAddr());
    auto* wqeScratchU64 = reinterpret_cast<__ubuf__ uint64_t*>(wqeScratch);
    for (uint32_t i = 0; i < ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE / sizeof(uint64_t); ++i) {
        wqeScratchU64[i] = 0;
    }

    auto* dstGlobal = reinterpret_cast<__gm__ uint8_t*>(dst);
    auto* srcGlobal = reinterpret_cast<__gm__ uint8_t*>(src);
    aclshmemx_udma_put_nbi<uint8_t>(dstGlobal, srcGlobal, wqeScratch, bytes, peer, kUdmaSyncId);
    aclshmemx_udma_quiet(peer);
}

extern "C" void LaunchUserBufferMtePut(
    void* stream, uint64_t fftsConfig, uint8_t* dst, uint8_t* src, uint32_t bytes, int32_t peer)
{
    UserBufferMtePutKernel<<<1, nullptr, stream>>>(fftsConfig, dst, src, bytes, peer);
}

extern "C" void LaunchUserBufferUdmaPut(void* stream, uint8_t* dst, uint8_t* src, uint32_t bytes, int32_t peer)
{
    UserBufferUdmaPutKernel<<<1, nullptr, stream>>>(dst, src, bytes, peer);
}
