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
constexpr uint64_t MESSAGE_SIZE = 64;

extern "C" __global__ __aicore__ void UDMAGetTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR dest_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        dest_addr = gva + peer * MESSAGE_SIZE;
        aclshmemx_udma_get_nbi(dest_addr, dest_addr, (__ubuf__ uint8_t *)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_get(uint32_t block_dim, void *stream, uint8_t *gva)
{
    UDMAGetTest<<<block_dim, nullptr, stream>>>(gva);
}

extern "C" __global__ __aicore__ void UDMAPutTest(GM_ADDR gva)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UB_ALIGN_SIZE * 2);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UB_ALIGN_SIZE * 2, 0);

    int64_t rank = aclshmem_my_pe();
    int64_t rank_size = aclshmem_n_pes();
    GM_ADDR src_addr;

    for (int64_t peer = 0; peer < rank_size; peer++) {
        if (peer == rank) {
            continue;
        }
        src_addr = gva + rank * MESSAGE_SIZE;
        aclshmemx_udma_put_nbi(src_addr, src_addr, (__ubuf__ uint8_t *)ubLocal.GetPhyAddr(), MESSAGE_SIZE, peer);
        aclshmemx_udma_quiet(peer);
    }
}

void test_udma_put(uint32_t block_dim, void *stream, uint8_t *gva)
{
    UDMAPutTest<<<block_dim, nullptr, stream>>>(gva);
}