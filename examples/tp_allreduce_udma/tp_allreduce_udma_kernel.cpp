/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tp_allreduce_udma_kernel.h"

#include "kernel_operator.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"

namespace {

constexpr uint32_t TP_SIZE = 2;
#if defined(ACLSHMEM_RELAY_SUPPORT)
constexpr uint32_t BOX_SIZE = 4;
constexpr uint32_t TAILCUT_PATH_COUNT = 3;
#endif
constexpr uint32_t REDUCE_TILE_BYTES = 16 * 1024;
constexpr uint32_t REDUCE_QUEUE_DEPTH = 1;
constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = 256;
constexpr int32_t PROF_FRAME_TOTAL = 0;
constexpr int32_t PROF_FRAME_REDUCE_SCATTER = 1;
constexpr int32_t PROF_FRAME_LOCAL_REDUCE = 2;
constexpr int32_t PROF_FRAME_ALLGATHER = 3;

ACLSHMEM_DEVICE void ProfStart(int32_t enabled, int32_t frameId)
{
    if (enabled != 0) {
        SHMEMI_PROF_START(frameId);
    }
}

ACLSHMEM_DEVICE void ProfEnd(int32_t enabled, int32_t frameId)
{
    if (enabled != 0) {
        SHMEMI_PROF_END(frameId);
    }
}

ACLSHMEM_DEVICE void SplitRange(
    uint32_t elements, uint32_t coreIdx, uint32_t coreCount, uint32_t& coreOffset, uint32_t& coreElements)
{
    uint32_t baseElements = elements / coreCount;
    uint32_t remainder = elements % coreCount;
    coreElements = baseElements + (coreIdx < remainder ? 1 : 0);
    coreOffset = coreIdx * baseElements + (coreIdx < remainder ? coreIdx : remainder);
}

#if defined(ACLSHMEM_RELAY_SUPPORT)
ACLSHMEM_DEVICE bool GetRelayPeers(int32_t myPe, int32_t targetPe, int32_t& relay0, int32_t& relay1)
{
    int32_t boxBase = myPe / static_cast<int32_t>(BOX_SIZE) * static_cast<int32_t>(BOX_SIZE);
    uint32_t relayCount = 0;
    for (uint32_t i = 0; i < BOX_SIZE; ++i) {
        int32_t candidate = boxBase + static_cast<int32_t>(i);
        if (candidate == myPe || candidate == targetPe) {
            continue;
        }
        if (relayCount == 0) {
            relay0 = candidate;
        } else if (relayCount == 1) {
            relay1 = candidate;
        }
        ++relayCount;
    }
    return relayCount == 2;
}

ACLSHMEM_DEVICE void GetTailcutSegment(
    uint32_t pathIdx, uint32_t totalElements, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight,
    uint32_t& offset, uint32_t& elementCount)
{
    uint64_t totalWeight =
        static_cast<uint64_t>(directWeight) + static_cast<uint64_t>(relay0Weight) + static_cast<uint64_t>(relay1Weight);
    if (totalWeight == 0) {
        offset = 0;
        elementCount = 0;
        return;
    }
    // Cumulative floor boundaries keep zero-weight paths empty while covering the complete range.
    uint32_t directEnd = static_cast<uint32_t>(static_cast<uint64_t>(totalElements) * directWeight / totalWeight);
    uint32_t relay0End = static_cast<uint32_t>(
        static_cast<uint64_t>(totalElements) * (static_cast<uint64_t>(directWeight) + relay0Weight) / totalWeight);
    if (pathIdx == 0) {
        offset = 0;
        elementCount = directEnd;
    } else if (pathIdx == 1) {
        offset = directEnd;
        elementCount = relay0End - directEnd;
    } else {
        offset = relay0End;
        elementCount = totalElements - offset;
    }
}
#endif

template <typename T>
ACLSHMEM_DEVICE void SendShardDirect(
    __gm__ T* remoteDst, __gm__ T* localSrc, __ubuf__ T* udmaBuffer, uint32_t elementCount, int32_t targetPe)
{
    aclshmemx_udma_put_nbi(remoteDst, localSrc, udmaBuffer, elementCount, targetPe);
    aclshmemx_udma_quiet(targetPe);
}

#if defined(ACLSHMEM_RELAY_SUPPORT)
template <typename T>
ACLSHMEM_DEVICE void SendShardTailcut(
    __gm__ T* remoteDst, __gm__ T* localSrc, __ubuf__ T* udmaBuffer, uint32_t elementCount, int32_t targetPe,
    int32_t myPe, uint32_t coreIdx, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    int32_t relay0 = -1;
    int32_t relay1 = -1;
    bool enableTailcut = GetRelayPeers(myPe, targetPe, relay0, relay1);

    if (!enableTailcut) {
        if (coreIdx == 0) {
            SendShardDirect(remoteDst, localSrc, udmaBuffer, elementCount, targetPe);
        }
        return;
    }

    if (coreIdx < TAILCUT_PATH_COUNT) {
        uint32_t offset = 0;
        uint32_t currentElements = 0;
        GetTailcutSegment(coreIdx, elementCount, directWeight, relay0Weight, relay1Weight, offset, currentElements);
        if (currentElements > 0) {
            if (coreIdx == 0) {
                aclshmemx_udma_put_nbi(remoteDst + offset, localSrc + offset, udmaBuffer, currentElements, targetPe);
            } else {
                int32_t relayPe = coreIdx == 1 ? relay0 : relay1;
                aclshmemx_udma_relay_put_nbi(
                    remoteDst + offset, localSrc + offset, udmaBuffer, currentElements, targetPe, relayPe);
            }
        }
    }

    // Every active-path NBI call must return before core0 starts scanning the completion slots.
    AscendC::SyncAll<true>();
    if (coreIdx == 0) {
        aclshmemx_udma_quiet(targetPe);
    }
}
#endif

template <typename T>
ACLSHMEM_DEVICE void SendShard(
    __gm__ T* remoteDst, __gm__ T* localSrc, __ubuf__ T* udmaBuffer, uint32_t elementCount, int32_t targetPe,
    int32_t myPe, uint32_t coreIdx, int32_t mode, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    if (mode == static_cast<int32_t>(TpAllReduceUdmaMode::BASELINE)) {
        if (coreIdx == 0) {
            SendShardDirect(remoteDst, localSrc, udmaBuffer, elementCount, targetPe);
        }
        return;
    }

#if defined(ACLSHMEM_RELAY_SUPPORT)
    SendShardTailcut(
        remoteDst, localSrc, udmaBuffer, elementCount, targetPe, myPe, coreIdx, directWeight, relay0Weight,
        relay1Weight);
#else
    (void)myPe;
    (void)directWeight;
    (void)relay0Weight;
    (void)relay1Weight;
#endif
}

template <typename T>
ACLSHMEM_DEVICE void ReduceTwoShards(
    __gm__ T* localShard, __gm__ T* peerShard, __gm__ T* outputShard, uint32_t shardElements, AscendC::TPipe& pipe)
{
    AscendC::GlobalTensor<T> localGm;
    AscendC::GlobalTensor<T> peerGm;
    AscendC::GlobalTensor<T> outputGm;
    localGm.SetGlobalBuffer(localShard);
    peerGm.SetGlobalBuffer(peerShard);
    outputGm.SetGlobalBuffer(outputShard);

    AscendC::TQue<AscendC::TPosition::VECIN, REDUCE_QUEUE_DEPTH> localQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, REDUCE_QUEUE_DEPTH> peerQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, REDUCE_QUEUE_DEPTH> outputQueue;
    pipe.InitBuffer(localQueue, REDUCE_QUEUE_DEPTH, REDUCE_TILE_BYTES);
    pipe.InitBuffer(peerQueue, REDUCE_QUEUE_DEPTH, REDUCE_TILE_BYTES);
    pipe.InitBuffer(outputQueue, REDUCE_QUEUE_DEPTH, REDUCE_TILE_BYTES);

    constexpr uint32_t tileElements = REDUCE_TILE_BYTES / sizeof(T);
    for (uint32_t offset = 0; offset < shardElements; offset += tileElements) {
        uint32_t currentElements = shardElements - offset;
        if (currentElements > tileElements) {
            currentElements = tileElements;
        }

        AscendC::DataCopyExtParams copyParams{};
        copyParams.blockCount = 1;
        copyParams.blockLen = currentElements * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        AscendC::LocalTensor<T> localUb = localQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> peerUb = peerQueue.AllocTensor<T>();
        AscendC::DataCopyPad(localUb, localGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(peerUb, peerGm[offset], copyParams, padParams);
        localQueue.EnQue(localUb);
        peerQueue.EnQue(peerUb);

        localUb = localQueue.DeQue<T>();
        peerUb = peerQueue.DeQue<T>();
        AscendC::LocalTensor<T> outputUb = outputQueue.AllocTensor<T>();
        AscendC::Add(outputUb, localUb, peerUb, currentElements);
        outputQueue.EnQue(outputUb);
        localQueue.FreeTensor(localUb);
        peerQueue.FreeTensor(peerUb);

        outputUb = outputQueue.DeQue<T>();
        AscendC::DataCopyPad(outputGm[offset], outputUb, copyParams);
        outputQueue.FreeTensor(outputUb);
    }
}

template <typename T>
ACLSHMEM_DEVICE void TpAllReduceUdmaImpl(
    uint64_t fftsConfig, GM_ADDR workspace, uint32_t elements, int teamId, int32_t perfMode, int32_t mode,
    int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    util_set_ffts_config(fftsConfig);

    aclshmem_team_t tpTeam = static_cast<aclshmem_team_t>(teamId);
    int32_t rankInTp = aclshmem_team_my_pe(tpTeam);
    int32_t tpSize = aclshmem_team_n_pes(tpTeam);
    if (tpSize != TP_SIZE || elements % TP_SIZE != 0) {
        return;
    }

    int32_t peerInTp = rankInTp ^ 1;
    int32_t myGlobalRank = -1;
#if defined(ACLSHMEM_RELAY_SUPPORT)
    if (mode == static_cast<int32_t>(TpAllReduceUdmaMode::TAILCUT)) {
        myGlobalRank = aclshmem_team_translate_pe(tpTeam, rankInTp, ACLSHMEM_TEAM_WORLD);
    }
#endif
    int32_t peerGlobalRank = aclshmem_team_translate_pe(tpTeam, peerInTp, ACLSHMEM_TEAM_WORLD);
    uint32_t shardElements = elements / TP_SIZE;
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t coreCount = AscendC::GetBlockNum();

    __gm__ T* input = reinterpret_cast<__gm__ T*>(workspace);
    __gm__ T* reduceScatterRecv = input + elements;
    __gm__ T* output = reduceScatterRecv + elements;

    __gm__ T* shardForPeer = input + peerInTp * shardElements;
    __gm__ T* peerRecvSlot = reduceScatterRecv + rankInTp * shardElements;
    __gm__ T* localShard = input + rankInTp * shardElements;
    __gm__ T* receivedPeerShard = reduceScatterRecv + peerInTp * shardElements;
    __gm__ T* reducedLocalShard = output + rankInTp * shardElements;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> udmaBuffer;
    pipe.InitBuffer(udmaBuffer, UDMA_WQE_SCRATCH_BYTES);
    AscendC::LocalTensor<uint8_t> udmaUb = udmaBuffer.GetWithOffset<uint8_t>(UDMA_WQE_SCRATCH_BYTES, 0);
    __ubuf__ T* udmaUbAddr = reinterpret_cast<__ubuf__ T*>(udmaUb.GetPhyAddr());

    TpAllReduceUdmaStage currentStage = static_cast<TpAllReduceUdmaStage>(stage);
    if (currentStage == TpAllReduceUdmaStage::REDUCE_SCATTER) {
        ProfStart(perfMode, PROF_FRAME_TOTAL);
        ProfStart(perfMode, PROF_FRAME_REDUCE_SCATTER);
        SendShard(
            peerRecvSlot, shardForPeer, udmaUbAddr, shardElements, peerGlobalRank, myGlobalRank, coreIdx, mode,
            directWeight, relay0Weight, relay1Weight);
        ProfEnd(perfMode, PROF_FRAME_REDUCE_SCATTER);
        return;
    }
    if (currentStage != TpAllReduceUdmaStage::LOCAL_REDUCE_ALLGATHER) {
        return;
    }

    ProfStart(perfMode, PROF_FRAME_LOCAL_REDUCE);
    // Split on reduce-tile boundaries so each core starts from an aligned GM offset.
    constexpr uint32_t reduceTileElements = REDUCE_TILE_BYTES / sizeof(T);
    uint32_t reduceTileCount = (shardElements + reduceTileElements - 1) / reduceTileElements;
    uint32_t coreTileOffset = 0;
    uint32_t coreTileCount = 0;
    SplitRange(reduceTileCount, coreIdx, coreCount, coreTileOffset, coreTileCount);
    uint32_t reduceOffset = coreTileOffset * reduceTileElements;
    uint32_t reduceElements = 0;
    if (coreTileCount > 0) {
        reduceElements = coreTileCount * reduceTileElements;
        if (reduceElements > shardElements - reduceOffset) {
            reduceElements = shardElements - reduceOffset;
        }
    }
    if (reduceElements > 0) {
        ReduceTwoShards(
            localShard + reduceOffset, receivedPeerShard + reduceOffset, reducedLocalShard + reduceOffset,
            reduceElements, pipe);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<true>();
    ProfEnd(perfMode, PROF_FRAME_LOCAL_REDUCE);

    ProfStart(perfMode, PROF_FRAME_ALLGATHER);
    SendShard(
        reducedLocalShard, reducedLocalShard, udmaUbAddr, shardElements, peerGlobalRank, myGlobalRank, coreIdx, mode,
        directWeight, relay0Weight, relay1Weight);
    ProfEnd(perfMode, PROF_FRAME_ALLGATHER);

    ProfEnd(perfMode, PROF_FRAME_TOTAL);
}

} // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tp_allreduce_udma_int32_t_kernel(
    uint64_t fftsConfig, GM_ADDR workspace, uint32_t elements, int teamId, int32_t perfMode, int32_t mode,
    int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    TpAllReduceUdmaImpl<int32_t>(
        fftsConfig, workspace, elements, teamId, perfMode, mode, stage, directWeight, relay0Weight, relay1Weight);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void tp_allreduce_udma_float16_t_kernel(
    uint64_t fftsConfig, GM_ADDR workspace, uint32_t elements, int teamId, int32_t perfMode, int32_t mode,
    int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    TpAllReduceUdmaImpl<float16_t>(
        fftsConfig, workspace, elements, teamId, perfMode, mode, stage, directWeight, relay0Weight, relay1Weight);
}

void launch_tp_allreduce_udma_int32_t(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, int32_t mode, int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    tp_allreduce_udma_int32_t_kernel<<<blockDim, nullptr, stream>>>(
        fftsConfig, workspace, elements, static_cast<int>(tpTeam), perfMode, mode, stage, directWeight, relay0Weight,
        relay1Weight);
}

void launch_tp_allreduce_udma_float16_t(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, int32_t mode, int32_t stage, uint32_t directWeight, uint32_t relay0Weight, uint32_t relay1Weight)
{
    tp_allreduce_udma_float16_t_kernel<<<blockDim, nullptr, stream>>>(
        fftsConfig, workspace, elements, static_cast<int>(tpTeam), perfMode, mode, stage, directWeight, relay0Weight,
        relay1Weight);
}
