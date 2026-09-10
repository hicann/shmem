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

extern "C" __global__ __aicore__ void copy_perf(GM_ADDR src, GM_ADDR res, uint32_t copypad_size, uint32_t copypad_times)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    AscendC::TPipe pipe;

    uint32_t cur_block_idx = AscendC::GetBlockIdx();
    uint32_t n_blocks = AscendC::GetBlockNum();

    uint32_t copy_block_step_size = copypad_size * copypad_times;

    uint64_t copy_ub;
    AscendC::LocalTensor<uint8_t> ubTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);
    AscendC::GlobalTensor<uint8_t> GmTensor;
    GmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src));
    GmTensor = GmTensor[copy_block_step_size * cur_block_idx];

    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64; // 64B for temporary buffer
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));

    uint64_t start_cycle, end_cycle;

    AscendC::DataCopyExtParams dataCopyParams(1, copypad_size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<uint8_t> padParams;

    start_cycle = AscendC::GetSystemCycle();
    for (size_t j = 0; j < copypad_times; j++) {
        AscendC::DataCopyPad(ubTensor, GmTensor[j * copypad_size], dataCopyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }
    end_cycle = AscendC::GetSystemCycle();

    uint32_t use_cycle = end_cycle - start_cycle;

    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(tmp_buff);
    ub_tensor.address_.dataLen = ub_size;

    aclshmemi_set_value((__gm__ uint8_t*)(res + cur_block_idx * sizeof(uint32_t)), use_cycle, ub_tensor, EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();
}

void copy_perf_kernel(
    uint32_t block_dim, void* stream, uint8_t* src, uint8_t* res, uint32_t copypad_size, uint32_t copypad_times)
{
    copy_perf<<<block_dim, nullptr, stream>>>(src, res, copypad_size, copypad_times);
}

// QP接口版本：按AIV均分预取区间，每个AIV用自身索引作为qp_idx提交CMO预取并等待完成
extern "C" __global__ __aicore__ void cmo_pretech(GM_ADDR src, uint32_t size)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }

    const auto cur_block_idx = AscendC::GetBlockIdx();
    const auto comm_block_dim = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    uint64_t base_per_core = size / comm_block_dim;
    uint64_t extra_bytes = size % comm_block_dim;
    uint64_t data_offset = 0;
    if (cur_block_idx < extra_bytes) {
        data_offset = cur_block_idx * (base_per_core + 1);
    } else {
        data_offset = extra_bytes * (base_per_core + 1) + (cur_block_idx - extra_bytes) * base_per_core;
    }
    if (cur_block_idx < extra_bytes) {
        base_per_core += 1;
    }
    if (base_per_core == 0) {
        return;
    }

    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64; // 64B for temporary buffer
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));

    aclshmemx_cmo_qp_nbi(
        reinterpret_cast<__gm__ uint8_t*>(src) + data_offset, static_cast<uint32_t>(base_per_core),
        ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, cur_block_idx, EVENT_ID0);
    aclshmemx_sdma_qp_quiet(tmp_buff, ub_size, cur_block_idx, EVENT_ID0);
}

void cmo_pretech_kernel(uint8_t* src, uint32_t size, void* stream) { cmo_pretech<<<1, nullptr, stream>>>(src, size); }

// 不带QP接口版本：固定使用QP 0，属于单核接口，仅由0号AIV执行
extern "C" __global__ __aicore__ void cmo_pretech_noqp(GM_ADDR src, uint32_t size)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    // Define temporary UB buffer
    constexpr uint32_t ub_offset = 1024;
    constexpr uint32_t ub_size = 64; // 64B for temporary buffer
    __ubuf__ uint8_t* tmp_buff = reinterpret_cast<__ubuf__ uint8_t*>(uint64_t(ub_offset));

    aclshmemx_cmo_nbi(
        reinterpret_cast<__gm__ uint8_t*>(src), size, ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH, tmp_buff, ub_size, EVENT_ID0);
    aclshmemx_sdma_quiet(tmp_buff, ub_size, EVENT_ID0);
}

void cmo_pretech_noqp_kernel(uint8_t* src, uint32_t size, void* stream)
{
    cmo_pretech_noqp<<<1, nullptr, stream>>>(src, size);
}
