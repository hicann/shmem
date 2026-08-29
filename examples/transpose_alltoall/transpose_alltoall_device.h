/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_ALLTOALL_DEVICE_H
#define TRANSPOSE_ALLTOALL_DEVICE_H

#include "kernel_operator.h"
#include "shmem.h"

#include "transpose_alltoall_kernel.hpp"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

struct TransposeAllToAllTiling {
    uint32_t B;
    uint32_t N;
    uint32_t S;
    uint32_t D;
    uint32_t rankSize;
};

namespace ShmemTransposeAllToAll {

template <uint32_t UB_STAGES_, class ElementInput_, class ElementOutput_>
class BlockTransposeBnsdToBsnd {
public:
    using ElementC = ElementInput_;
    using ElementD = ElementOutput_;

    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t MAX_GROUPS_PER_BURST = 4095;

    struct Params {
        uint32_t D{0};
        uint32_t N{0};
        uint32_t S{0};

        ACLSHMEM_DEVICE Params() {}
        ACLSHMEM_DEVICE Params(uint32_t d, uint32_t n, uint32_t s) : D(d), N(n), S(s) {}
    };

    ACLSHMEM_DEVICE
    BlockTransposeBnsdToBsnd(Params const& params = Params{}) : params_(params)
    {
        static_assert(
            sizeof(ElementC) == sizeof(ElementD), "in/out element size must match for in-place transpose UB sizing");
        uint32_t ubBytes = AscendC::GetUBSizeInBytes();
        uint32_t bytesPerElem = sizeof(ElementD);
        tileElements_ = ubBytes / (UB_STAGES * bytesPerElem);
        tileElements_ = (tileElements_ / 32) * 32;

        uint32_t ubOffset = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubBufList_[i] = AscendC::LocalTensor<ElementD>(AscendC::TPosition::VECCALC, ubOffset, tileElements_);
            ubOffset += tileElements_ * sizeof(ElementD);
        }
    }

    ACLSHMEM_DEVICE void InitBlockLoop()
    {
        int32_t evMTE2MTE3 = 0, evMTE3MTE2 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            evMTE2MTE3_[i] = evMTE2MTE3++;
            evMTE3MTE2_[i] = evMTE3MTE2++;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2_[i]);
        }
    }

    ACLSHMEM_DEVICE void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2_[i]);
        }
    }

    ACLSHMEM_DEVICE void operator()(
        AscendC::GlobalTensor<ElementC>& gmInput, AscendC::GlobalTensor<ElementD>& gmOutput, uint32_t blockSize,
        uint64_t globalOffset)
    {
        uint32_t D = params_.D;
        uint32_t N = params_.N;
        uint32_t S = params_.S;
        uint64_t outStrideB = static_cast<uint64_t>(S) * N * D;
        uint64_t outStrideS = static_cast<uint64_t>(N) * D;

        uint64_t b = globalOffset / outStrideB;
        uint64_t rem = globalOffset - b * outStrideB;
        uint64_t s = rem / outStrideS;
        uint64_t n = (rem - s * outStrideS) / D;

        uint32_t numGroups = blockSize / D;

        InitBlockLoop();

        uint64_t curB = b, curS = s, curN = n;
        uint32_t groupsProcessed = 0;
        uint32_t tileIdx = 0;

        while (groupsProcessed < numGroups) {
            uint32_t groupsThisTile = Min<uint32_t>(numGroups - groupsProcessed, N - curN);
            groupsThisTile = Min<uint32_t>(groupsThisTile, MAX_GROUPS_PER_BURST);

            uint32_t dOff = 0;
            while (dOff < D) {
                uint32_t chunkElems = Min<uint32_t>(tileElements_ / groupsThisTile, D - dOff);
                if (chunkElems == 0) {
                    chunkElems = 1;
                }
                uint32_t stageId = tileIdx % UB_STAGES;
                CopyInByBSND(stageId, gmInput, curB, curS, curN, groupsThisTile, dOff, chunkElems);
                CopyOut(stageId, gmOutput, groupsThisTile, groupsProcessed, dOff, chunkElems);

                dOff += chunkElems;
                ++tileIdx;
            }

            groupsProcessed += groupsThisTile;
            curN += groupsThisTile;
            if (curN >= N) {
                curN -= N;
                ++curS;
                if (curS >= S) {
                    curS -= S;
                    ++curB;
                }
            }
        }

        FinalizeBlockLoop();
    }

private:
    ACLSHMEM_DEVICE void CopyInByBSND(
        uint32_t stageId, AscendC::GlobalTensor<ElementC>& gmInput, uint64_t startB, uint64_t startS, uint64_t startN,
        uint32_t numGroups, uint32_t dOff, uint32_t chunkElems)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2_[stageId]);
        auto& ubBuf = ubBufList_[stageId];

        uint32_t D = params_.D;
        uint32_t N = params_.N;
        uint32_t S = params_.S;

        uint64_t srcOff = startB * N * S * D + startN * S * D + startS * D + dOff;
        uint64_t srcStride64 = (static_cast<uint64_t>(S) * D - chunkElems) * sizeof(ElementC);
        if (srcStride64 > UINT32_MAX)
            return;
        uint32_t srcStride = static_cast<uint32_t>(srcStride64);
        AscendC::DataCopyExtParams srcParams(
            static_cast<uint16_t>(numGroups), chunkElems * sizeof(ElementC), srcStride, 0, 0);
        AscendC::DataCopyPadExtParams<ElementC> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad<ElementC, AscendC::PaddingMode::Compact>(ubBuf, gmInput[srcOff], srcParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2MTE3_[stageId]);
    }

    ACLSHMEM_DEVICE void CopyOut(
        uint32_t stageId, AscendC::GlobalTensor<ElementD>& gmOutput, uint32_t numGroups, uint32_t groupOffset,
        uint32_t dOff, uint32_t chunkElems)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMTE2MTE3_[stageId]);
        auto& ubBuf = ubBufList_[stageId];

        uint32_t D = params_.D;
        uint32_t dstIdx = groupOffset * D + dOff;
        uint64_t dstStride64 = (static_cast<uint64_t>(D - chunkElems) * sizeof(ElementD));
        if (dstStride64 > UINT32_MAX)
            return;
        uint32_t dstStride = static_cast<uint32_t>(dstStride64);
        AscendC::DataCopyExtParams dstParams(
            static_cast<uint16_t>(numGroups), chunkElems * sizeof(ElementD), 0, dstStride, 0);
        AscendC::DataCopyPad<ElementD, AscendC::PaddingMode::Compact>(gmOutput[dstIdx], ubBuf, dstParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3MTE2_[stageId]);
    }

    Params params_;
    uint32_t tileElements_{0};

    AscendC::LocalTensor<ElementD> ubBufList_[UB_STAGES];
    int32_t evMTE2MTE3_[UB_STAGES];
    int32_t evMTE3MTE2_[UB_STAGES];
};

template <class Element_, uint32_t UB_STAGES_ = 2>
class BlockAllToAllRemoteCopy {
public:
    using Element = Element_;

    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    ACLSHMEM_DEVICE
    BlockAllToAllRemoteCopy()
    {
        uint32_t ubBytes = AscendC::GetUBSizeInBytes();
        stageBytes_ = (ubBytes / UB_STAGES / 32) * 32;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubBufList_[i] = AscendC::LocalTensor<Element>(
                AscendC::TPosition::VECCALC, i * stageBytes_, stageBytes_ / sizeof(Element));
        }
    }

    ACLSHMEM_DEVICE void InitBlockLoop()
    {
        stageIdx_ = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            eventId_[i] = static_cast<int32_t>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId_[i]);
        }
    }

    ACLSHMEM_DEVICE void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId_[i]);
        }
    }

    ACLSHMEM_DEVICE void operator()(
        __gm__ Element* remoteSymmetricSrc, __gm__ Element* localDst, uint32_t elemCount, int32_t pe)
    {
        uint32_t stageId = stageIdx_ % UB_STAGES;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId_[stageId]);

        AscendC::GlobalTensor<Element> srcTensor;
        srcTensor.SetGlobalBuffer(remoteSymmetricSrc);
        AscendC::GlobalTensor<Element> dstTensor;
        dstTensor.SetGlobalBuffer(localDst);
        aclshmemx_mte_get_nbi(
            dstTensor, srcTensor, ubBufList_[stageId], elemCount, pe, static_cast<uint32_t>(eventId_[stageId]));

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId_[stageId]);
        ++stageIdx_;
    }

private:
    uint32_t stageBytes_{0};
    uint32_t stageIdx_{0};
    AscendC::LocalTensor<Element> ubBufList_[UB_STAGES];
    int32_t eventId_[UB_STAGES];
};

} // namespace ShmemTransposeAllToAll

template <class ElementInput, class ElementOutput>
ACLSHMEM_DEVICE void TransposeAllToAllImpl(
    uint32_t B, uint32_t N, uint32_t S, uint32_t D, GM_ADDR gmInput, GM_ADDR gmOutput, GM_ADDR symmetricPtr,
    uint32_t rank, uint32_t rankSize)
{
    constexpr uint32_t KERNEL_WORKSPACE_STAGES = 4;
    constexpr uint32_t UB_STAGES = 2;

    using BlockTranspose = ShmemTransposeAllToAll::BlockTransposeBnsdToBsnd<UB_STAGES, ElementInput, ElementOutput>;
    using BlockAllToAll = ShmemTransposeAllToAll::BlockAllToAllRemoteCopy<ElementOutput, UB_STAGES>;
    using TransposeAllToAllKernel =
        ShmemTransposeAllToAll::TransposeAllToAll<BlockTranspose, BlockAllToAll, KERNEL_WORKSPACE_STAGES>;

    typename TransposeAllToAllKernel::Params params{B, N, S, D, rank, rankSize, gmInput, gmOutput, symmetricPtr};
    TransposeAllToAllKernel transposeAllToAllKernel;
    transposeAllToAllKernel(params);
}

// Global kernel entry
template <class ElementInput, class ElementOutput>
__global__ __mix__(1, 2) __schedmode__(1) void Ascend950TransposeAllToAll(
    uint64_t fftsAddr, GM_ADDR gmInput, GM_ADDR gmOutput, GM_ADDR symmetricPtr, TransposeAllToAllTiling tiling)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    if (tiling.B == 0 || tiling.N == 0 || tiling.S == 0 || tiling.D == 0)
        return;
    uint32_t B = tiling.B;
    uint32_t N = tiling.N;
    uint32_t S = tiling.S;
    uint32_t D = tiling.D;
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    TransposeAllToAllImpl<ElementInput, ElementOutput>(B, N, S, D, gmInput, gmOutput, symmetricPtr, rank, rankSize);
}

#endif // TRANSPOSE_ALLTOALL_DEVICE_H
