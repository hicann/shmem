/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#if defined(ENABLE_ASCENDC_DUMP)
#include "host/utils/debug.h"
#endif

// shmem_host
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
#include "catcoc/dgemm/kernel/allgather_matmul.h"

static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;

#if defined(ENABLE_ASCENDC_DUMP)
CATLASS_GLOBAL
void ShmemAllGatherMatmul(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k, GM_ADDR dump)
{
    AscendC::InitDump(false, dump, ALL_DUMPSIZE);
#else
CATLASS_GLOBAL
void ShmemAllGatherMatmul(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k)
{
#endif
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    // Prepare comm address
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * rankSize, n};

    // Block level, define BlockMmad
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
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    constexpr uint32_t SWIZZLE_GROUP_SIZE = 7;
    constexpr uint32_t SWIZZLE_DIRECTION = 1;
    using BlockMmadScheduler = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<SWIZZLE_GROUP_SIZE,
                                                                                            SWIZZLE_DIRECTION>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr uint32_t COMM_BLOCK_ROWS = 64;
    constexpr uint32_t COMM_BLOCK_COLUMNS_DIVISOR = 2;
    constexpr uint32_t CORE_SPLIT_ROWS = 20;
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

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmul<
        BlockMmad,
        BlockEpilogueAllGather,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    typename BlockEpilogueAllGather::Params allGatherParams{};

    // Prepare params
    typename AllGatherMatmulKernel::Params params{
        problemShape,
        rank, rankSize,
        COMM_INTERVAL,
        gmA, layoutA,
        gmB, layoutB,
        gmC, layoutC,
        gmSymmetric,
        allGatherParams
    };

    // Call kernel
    AllGatherMatmulKernel matmulCommKernel;
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

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    // Kernel-need params parse
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    int n_ranks = options.rankSize;
    int rank_id = options.rankId;
    std::string ipport = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t device_id = options.deviceIdList[rank_id];

    // Acl && Shmem init
    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);

    shmem_init_attr_t *attributes;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport.c_str(), &attributes);
    status = shmem_init_attr(SHMEMX_INIT_WITH_DEFAULT, attributes);

    // ACLStream init
    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);

    std::cout << "[TEST] input rank_size: " << n_ranks << " rank_id:" << rank_id << std::endl;

    // status = shmem_set_conf_store_tls(false, nullptr, 0);

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * n_ranks * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rank_id) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rank_id) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&cHost), cSize));

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling AG_MM kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        uint64_t fftsAddr = shmemx_get_ffts_config();
#if defined(ENABLE_ASCENDC_DUMP)
        uint8_t *deviceDump{nullptr};
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceDump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, cDevice, gmSymmetric,
            m, n, k, deviceDump);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        Adx::AdumpPrintWorkSpace(deviceDump, ALL_DUMPSIZE, stream, "test");
#else
        ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr,
            aDevice, bDevice, cDevice, gmSymmetric,
            m, n, k);
#endif
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling AG_MM kernel " << std::endl;

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.GetDataPath("shmem_output.bin"), cHost, cSize);
    if (rank_id == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));

    status = aclrtDestroyStream(stream);

    status = shmem_finalize();
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    return 0;
}