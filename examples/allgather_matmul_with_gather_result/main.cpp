/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// aclshmem_host
#include "host/aclshmem_host_def.h"
#include "host/mem/aclshmem_host_heap.h"
#include "host/init/aclshmem_host_init.h"
#include "host/data_plane/aclshmem_host_rma.h"
#include "host/team/aclshmem_host_team.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.h"
#include "catcoc/comm_epilogue/comm_dispatch_policy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.h"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.h"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.h"
#include "catcoc/detail/remote_copy_type.h"
#include "catcoc/dgemm/block/block_swizzle_allgather.h"
#include "catcoc/dgemm/kernel/allgather_matmul_with_gather_result.h"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementGatherA = ElementA;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutGatherA = LayoutA;

CATLASS_GLOBAL
void ShmemAllGatherMatmulWithGatherResult(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmGatherA, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k
)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    // Prepare comm address
    uint32_t rank = aclshmem_my_pe();
    uint32_t rankSize = aclshmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * rankSize, n};
    LayoutGatherA layoutGatherA{m * rankSize, k};

    constexpr bool ENABLE_UNIT_FLAG = true;
    constexpr int L1TILEM = 128;
    constexpr int L1TILEN = 256;
    constexpr int L1TILEK = 256;
    constexpr int L0TILEM = 128;
    constexpr int L0TILEN = 256;
    constexpr int L0TILEK = 64;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
    using L1TileShape = Catlass::GemmShape<L1TILEM, L1TILEN, L1TILEK>;
    using L0TileShape = Catlass::GemmShape<L0TILEM, L0TILEN, L0TILEK>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using GatherAType = AType;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy,
        L1TileShape, L0TileShape,
        AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockScheduler = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<SWIZZLE_GROUP_SIZE,
                                                                                        SWIZZLE_DIRECTION>;
    using BlockCommScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;
    using BlockCopyGatherAScheduler = CommEpilogue::Block::BlockSchedulerCopyGatherA;

    using RemoteSrcType = AType;
    using RemoteDstType = GatherAType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS_DIVISOR = 2;
    constexpr uint32_t CORE_SPLIT_ROWS = 16;
    constexpr uint32_t CORE_SPLIT_COLUMNS = 1;
    using CommBlockShape = Catlass::MatrixShape<COMM_BLOCK_ROWS, UINT_MAX / COMM_BLOCK_COLUMNS_DIVISOR>;
    using CommCoreSplit = Catlass::MatrixShape<CORE_SPLIT_ROWS, CORE_SPLIT_COLUMNS>;

    constexpr uint32_t UB_STAGES = 2;
    constexpr uint32_t ALLGATHER_TILE_ROWS = 32;
    constexpr uint32_t ALLGATHER_TILE_COLUMNS = 256;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<ALLGATHER_TILE_ROWS, ALLGATHER_TILE_COLUMNS>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueAllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t GATHER_BLOCK_ROWS = 48;
    constexpr uint32_t GATHER_BLOCK_COLUMNS_DIVISOR = 2;
    constexpr uint32_t GATHER_TILE_ROWS = 48;
    constexpr uint32_t GATHER_TILE_COLUMNS = 1024;
    using CopyGatherABlockShape = Catlass::MatrixShape<GATHER_BLOCK_ROWS, UINT_MAX / GATHER_BLOCK_COLUMNS_DIVISOR>;
    using CopyGatherATileShape = Catlass::MatrixShape<GATHER_TILE_ROWS, GATHER_TILE_COLUMNS>;
    using CopyGatherADispatchPolicy = CommEpilogue::EpilogueAtlasA2CommLocalCopy<UB_STAGES>;
    using BlockEpilogueCopyGatherA = CommEpilogue::Block::CommBlockEpilogue<
        CopyGatherADispatchPolicy,
        AType, GatherAType,
        CopyGatherABlockShape,
        CopyGatherATileShape,
        TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulWithGatherResultKernel = DGemm::Kernel::AllGatherMatmulWithGatherResult<
        BlockMmad,
        BlockEpilogueAllGather,
        BlockEpilogueCopyGatherA,
        BlockScheduler,
        BlockCommScheduler,
        BlockCopyGatherAScheduler,
        WORKSPACE_STAGES
    >;

    typename BlockEpilogueAllGather::Params allGatherParams{};
    typename BlockEpilogueCopyGatherA::Params copyGatherAParams{};

    // Prepare params
    typename AllGatherMatmulWithGatherResultKernel::Params params{
        problemShape,
        rank, rankSize,
        gmA, layoutA,
        gmB, layoutB,
        gmSymmetric,
        allGatherParams,
        copyGatherAParams,
        gmGatherA, layoutGatherA,
        gmC, layoutC,
        COMM_INTERVAL
    };

    AllGatherMatmulWithGatherResultKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul rank_size rank_id ip_port m n k [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        dataPath = argv[DATA_PATH_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }

    std::string GetDataPath(std::string const &fileName = "") const
    {
        return dataPath + "/" + fileName;
    }
};
static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};
aclshmemx_uniqueid_t default_flag_uid;
int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    return ACLSHMEM_SUCCESS;
}
int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;

    // Kernel-need params parse
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    int n_ranks = options.rankSize;
    int rank_id = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t device_id = options.deviceIdList[rank_id];

    // Acl && Shmem init
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, ipPort.c_str(), &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    // ACLStream init
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    std::cout << "[TEST] input rank_size: " << n_ranks << " rank_id:" << rank_id << std::endl;

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * n_ranks * n * sizeof(__fp16);
    size_t gatherASize = static_cast<size_t>(m) * n_ranks * k * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rank_id) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rank_id) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&cHost), cSize));

    uint8_t *gatherADevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&gatherADevice), gatherASize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *gatherAHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&gatherAHost), gatherASize));

    void *symmPtr = aclshmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling AG_MM kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        uint64_t fftsAddr = util_get_ffts_config();
        ShmemAllGatherMatmulWithGatherResult<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, cDevice, gatherADevice, gmSymmetric,
            m, n, k
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling AG_MM kernel " << std::endl;

    if (rank_id == 0) {
        ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
        ACL_CHECK(aclrtMemcpy(gatherAHost, gatherASize, gatherADevice, gatherASize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(options.GetDataPath("aclshmem_output.bin"), cHost, cSize);
        WriteFile(options.GetDataPath("aclshmem_gather_a.bin"), gatherAHost, gatherASize);
        std::printf("test finished\n");
    }

    aclshmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFreeHost(gatherAHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFree(gatherADevice));

    status = aclrtDestroyStream(stream);

    status = aclshmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    return 0;
}
