#ifndef TRANSPOSE_ALLTOALL_EXAMPLE_KERNEL_HPP
#define TRANSPOSE_ALLTOALL_EXAMPLE_KERNEL_HPP

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

namespace ShmemTransposeAllToAll {

template <typename T>
ACLSHMEM_DEVICE T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
ACLSHMEM_DEVICE T CeilDiv(T a, T b)
{
    return (a + b - 1) / b;
}

template <class BlockTranspose_, class BlockAllToAll_, uint32_t WORKSPACE_STAGES_>
class TransposeAllToAll {
public:
    using BlockTranspose = BlockTranspose_;
    using ElementInput = typename BlockTranspose::ElementC;
    using ElementOutput = typename BlockTranspose::ElementD;

    using BlockAllToAll = BlockAllToAll_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t TILE_M = 1024 * 1024;
    static constexpr uint32_t COMM_SUB_BLOCK = 32 * 1024;

    static constexpr int64_t IPC_BUFF_MAX_SIZE = 1024UL * 1024UL * 1000;
    static constexpr uint32_t PING_PONG_SIZE = 2;
    static constexpr uint32_t ACLSHMEM_DATA_CACHE_LINE_SIZE = 64;
    static constexpr uint32_t SYNC_UNIT_STRIDE_I32 = ACLSHMEM_DATA_CACHE_LINE_SIZE / sizeof(int32_t);
    static constexpr size_t IPC_BUFF_HALF_SIZE = IPC_BUFF_MAX_SIZE / PING_PONG_SIZE;

    struct Params {
        uint32_t B;
        uint32_t N;
        uint32_t S;
        uint32_t D;

        uint32_t rankIdx;
        uint32_t rankSize;

        __gm__ ElementInput* ptrInput;
        __gm__ ElementOutput* ptrOutput;

        GM_ADDR ptrSymmetric;

        ACLSHMEM_DEVICE
        Params() {}

        ACLSHMEM_DEVICE
        Params(
            uint32_t B_, uint32_t N_, uint32_t S_, uint32_t D_, uint32_t rankIdx_, uint32_t rankSize_,
            GM_ADDR ptrInput_, GM_ADDR ptrOutput_, GM_ADDR ptrSymmetric_)
            : B(B_),
              N(N_),
              S(S_),
              D(D_),
              rankIdx(rankIdx_),
              rankSize(rankSize_),
              ptrInput(reinterpret_cast<__gm__ ElementInput*>(ptrInput_)),
              ptrOutput(reinterpret_cast<__gm__ ElementOutput*>(ptrOutput_)),
              ptrSymmetric(ptrSymmetric_)
        {}
    };

    ACLSHMEM_DEVICE
    TransposeAllToAll() {}

    template <int32_t CORE_TYPE = g_coreType>
    ACLSHMEM_DEVICE void operator()(Params& params);

    // AIC: no-op
    template <>
    ACLSHMEM_DEVICE void operator()<AscendC::AIC>(Params& params)
    {
        (void)params;
    }

    // AIV: main execution
    template <>
    ACLSHMEM_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        uint32_t B = params.B;
        uint32_t N = params.N;
        uint32_t S = params.S;
        uint32_t D = params.D;

        uint64_t localElements = static_cast<uint64_t>(B) * N * S * D;
        uint64_t chunkSize = localElements / params.rankSize;

        uint32_t commSizeM = static_cast<uint32_t>(Min<uint64_t>(TILE_M, chunkSize));
        commSizeM = (commSizeM / D) * D;
        if (commSizeM == 0)
            commSizeM = D;

        uint64_t workspaceBytes =
            static_cast<uint64_t>(WORKSPACE_STAGES) * aicoreNum * commSizeM * sizeof(ElementOutput);
        if (workspaceBytes > static_cast<uint64_t>(IPC_BUFF_HALF_SIZE))
            return;

        uint64_t blocksPerChunk64 = CeilDiv<uint64_t>(chunkSize, commSizeM);
        uint64_t totalBlocks64 = blocksPerChunk64 * params.rankSize;
        if (totalBlocks64 > static_cast<uint64_t>(INT32_MAX))
            return;

        uint32_t totalBlocks = static_cast<uint32_t>(totalBlocks64);
        uint32_t totalSteps = (CeilDiv<uint32_t>(totalBlocks, aicoreNum));

        uint32_t subBlocks = CeilDiv<uint32_t>(commSizeM, COMM_SUB_BLOCK);
        uint32_t tasksPerSlot = subBlocks * params.rankSize;

        uint32_t perSlotBytes = commSizeM * sizeof(ElementOutput);
        uint32_t perStageBytes = aicoreNum * perSlotBytes;

        auto syncTransposeFinish = reinterpret_cast<__gm__ int32_t*>(params.ptrSymmetric + IPC_BUFF_HALF_SIZE);
        auto syncCommFinish = syncTransposeFinish + aicoreNum * SYNC_UNIT_STRIDE_I32;

        if (subcoreIdx == 0 && aicoreIdx == 0) {
            for (uint32_t i = 0; i < aicoreNum; ++i) {
                aclshmemx_signal_op(
                    syncTransposeFinish + i * SYNC_UNIT_STRIDE_I32, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
                aclshmemx_signal_op(syncCommFinish + i * SYNC_UNIT_STRIDE_I32, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }
        aclshmemx_barrier_all_vec();

        BlockTranspose transpose(typename BlockTranspose::Params{D, N, S});
        BlockAllToAll allToAll;

        if (subcoreIdx == 1) {
            // ====== TRANSPOSE ======
            for (uint32_t blockIdx = aicoreIdx; blockIdx < totalBlocks; blockIdx += aicoreNum) {
                uint32_t chunkIdx = blockIdx % params.rankSize;
                uint32_t blockIdxInChunk = blockIdx / params.rankSize;
                uint32_t step = blockIdx / aicoreNum;
                uint32_t stageId = step % WORKSPACE_STAGES;
                uint64_t blockOffsetInChunk = blockIdxInChunk * commSizeM;
                uint32_t actualBlockSize =
                    static_cast<uint32_t>(Min<uint64_t>(commSizeM, chunkSize - blockOffsetInChunk));

                auto curSymmetric = params.ptrSymmetric + stageId * perStageBytes;
                uint64_t srcOffset = chunkIdx * chunkSize + blockOffsetInChunk;

                if (step >= WORKSPACE_STAGES) {
                    for (uint32_t rankId = 0; rankId < params.rankSize; ++rankId) {
                        auto remoteSyncComm = static_cast<__gm__ int32_t*>(shmem_ptr(syncCommFinish, rankId));
                        for (uint32_t commCoreIdx = 0; commCoreIdx < aicoreNum; ++commCoreIdx) {
                            aclshmem_signal_wait_until(
                                remoteSyncComm + commCoreIdx * SYNC_UNIT_STRIDE_I32, ACLSHMEM_CMP_GE,
                                step + 1 - WORKSPACE_STAGES);
                        }
                    }
                }

                __gm__ ElementOutput* dstPtr =
                    reinterpret_cast<__gm__ ElementOutput*>(curSymmetric) + aicoreIdx * commSizeM;

                AscendC::GlobalTensor<ElementInput> gmInput;
                gmInput.SetGlobalBuffer(params.ptrInput);
                AscendC::GlobalTensor<ElementOutput> gmOutput;
                gmOutput.SetGlobalBuffer(dstPtr);
                transpose(gmInput, gmOutput, actualBlockSize, srcOffset);

                aclshmem_fence();
                aclshmemx_signal_op(
                    syncTransposeFinish + aicoreIdx * SYNC_UNIT_STRIDE_I32, step + 1, ACLSHMEM_SIGNAL_SET,
                    params.rankIdx);
            }

        } else {
            // ====== COMM (cross-core parallel pulls) ======
            for (uint32_t step = 0; step < totalSteps; ++step) {
                uint32_t stageId = step % WORKSPACE_STAGES;
                auto curSymmetric = params.ptrSymmetric + stageId * perStageBytes;

                uint32_t validSlots = Min<uint32_t>(aicoreNum, totalBlocks - step * aicoreNum);
                uint32_t totalTasks = validSlots * tasksPerSlot;

                allToAll.InitBlockLoop();

                for (uint32_t taskIdx = aicoreIdx; taskIdx < totalTasks; taskIdx += aicoreNum) {
                    int32_t srcRankIdx = static_cast<int32_t>(taskIdx % params.rankSize);
                    uint32_t subIdx = (taskIdx / params.rankSize) % subBlocks;
                    uint32_t slot = taskIdx / tasksPerSlot;
                    uint32_t blockIdx = step * aicoreNum + slot;
                    uint32_t chunkIdx = blockIdx % params.rankSize;
                    if (chunkIdx != params.rankIdx) {
                        continue;
                    }
                    uint32_t blockIdxInChunk = blockIdx / params.rankSize;
                    uint64_t blockOffsetInChunk = blockIdxInChunk * commSizeM;
                    uint32_t actualBlockSize =
                        static_cast<uint32_t>(Min<uint64_t>(commSizeM, chunkSize - blockOffsetInChunk));
                    uint32_t subOffset = subIdx * COMM_SUB_BLOCK;
                    if (subOffset >= actualBlockSize) {
                        continue;
                    }
                    uint32_t actualSubBlockSize = Min<uint32_t>(COMM_SUB_BLOCK, actualBlockSize - subOffset);
                    auto remoteSyncTranspose = static_cast<__gm__ int32_t*>(shmem_ptr(syncTransposeFinish, srcRankIdx));
                    aclshmem_signal_wait_until(
                        remoteSyncTranspose + slot * SYNC_UNIT_STRIDE_I32, ACLSHMEM_CMP_GE, step + 1);

                    __gm__ ElementOutput* remoteSymmetricSrc =
                        reinterpret_cast<__gm__ ElementOutput*>(curSymmetric) + slot * commSizeM + subOffset;
                    __gm__ ElementOutput* localDst =
                        params.ptrOutput + srcRankIdx * chunkSize + blockOffsetInChunk + subOffset;

                    allToAll(remoteSymmetricSrc, localDst, actualSubBlockSize, srcRankIdx);
                }

                allToAll.FinalizeBlockLoop();
                aclshmem_fence();
                aclshmemx_signal_op(
                    syncCommFinish + aicoreIdx * SYNC_UNIT_STRIDE_I32, step + 1, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }

        aclshmemx_barrier_all_vec();

        if (subcoreIdx == 0 && aicoreIdx == 0) {
            for (uint32_t i = 0; i < aicoreNum; ++i) {
                aclshmemx_signal_op(
                    syncTransposeFinish + i * SYNC_UNIT_STRIDE_I32, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
                aclshmemx_signal_op(syncCommFinish + i * SYNC_UNIT_STRIDE_I32, 0, ACLSHMEM_SIGNAL_SET, params.rankIdx);
            }
        }
        aclshmemx_barrier_all_vec();
    }
};

} // namespace ShmemTransposeAllToAll

#endif // TRANSPOSE_ALLTOALL_EXAMPLE_KERNEL_HPP
