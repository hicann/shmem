/**
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
#include <algorithm>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
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
#include "catcoc/dgemm/kernel/matmul_reduce_scatter_padding.h"

using namespace AscendC;
using namespace Catcoc;

constexpr size_t NPU_MALLOC_SPACE = 1024UL * 1024 * 1024;

constexpr uint32_t BLOCK_NUM = 20;
constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 256;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementD = half;

template <bool PADDING_A, bool PADDING_B>
CATLASS_GLOBAL
void ShmemMatmulReduceScatterPadding(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR gmWA, GM_ADDR gmWB,
    GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k
)
{
    util_set_ffts_config(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;

    uint32_t rankIdx = aclshmem_my_pe();
    uint32_t rankSize = aclshmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m / rankSize, n};

    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(ElementA);

    bool isNeedPaddingA = Catcoc::Padding::IsNeedPadding(layoutA, alignByElement);
    bool isNeedPaddingB = Catcoc::Padding::IsNeedPadding(layoutB, alignByElement);
    
    using PaddingHelperA = typename Catcoc::Padding::PaddingHelper<AType, PADDING_A>;
    using LayoutWA = typename PaddingHelperA::LayoutW;
    LayoutWA layoutWA = PaddingHelperA::GetLayoutW(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
    using ActualTypeA = typename PaddingHelperA::ActualType;
    using GlobalPaddingA = typename PaddingHelperA::GlobalPadding;
    
    using PaddingHelperB = typename Catcoc::Padding::PaddingHelper<BType, PADDING_B>;
    using LayoutWB = typename PaddingHelperB::LayoutW;
    LayoutWB layoutWB = PaddingHelperB::GetLayoutW(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
    using ActualTypeB = typename PaddingHelperB::ActualType;
    using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, ActualTypeA, ActualTypeB, CType
    >;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0, true>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, 256>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t UB_STAGES = 2;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<32, 256>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
        Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueReduceScatterDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 10;
    using MatmulReduceScatterKernel = DGemm::Kernel::MatmulReduceScatterPadding<
        GlobalPaddingA,
        GlobalPaddingB,
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};

    typename MatmulReduceScatterKernel::Params params{
        problemShape, rankIdx, rankSize,
        COMM_INTERVAL,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        gmWA, layoutWA,
        gmWB, layoutWB,
        gmSymmetric,
        reduceScatterParams
    };

    MatmulReduceScatterKernel matmulReduceScatterKernel;
    matmulReduceScatterKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: matmul_reduce_scatter_padding rank_size rank_id ip_port m n k [device_id_list]\n";

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

    std::cout << "[TEST] input rank_size: " << n_ranks << " rank_id:" << rank_id << " input_ip: " << ipPort << "\n";

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

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(ElementA);

    bool isNeedPaddingA = Catcoc::Padding::IsNeedPadding(layoutA, alignByElement);
    bool isNeedPaddingB = Catcoc::Padding::IsNeedPadding(layoutB, alignByElement);

    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, Catlass::layout::ColumnMajor> &&
                           std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>,
                           Catlass::GemmShape<TILE_N, TILE_M, TILE_K>, Catlass::GemmShape<TILE_M, TILE_N, TILE_K>>;

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t dSize = static_cast<size_t>(m) * n * sizeof(__fp16);
    size_t dSizeScatter = dSize / n_ranks;

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

    uint8_t *dDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&dDevice), dSizeScatter, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dHost;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&dHost), dSize));

    size_t sizeWA = Catcoc::Padding::GetWorkspaceLen(layoutA, L1TileShape::M, L1TileShape::K) * sizeof(ElementA);
    size_t sizeWB = Catcoc::Padding::GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(ElementB);

    uint8_t *waDevice{nullptr};
    if (isNeedPaddingA) {
        aclrtMalloc(reinterpret_cast<void **>(&waDevice), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        waDevice = aDevice;
    }

    uint8_t *wbDevice{nullptr};
    if (isNeedPaddingB) {
        aclrtMalloc(reinterpret_cast<void **>(&wbDevice), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        wbDevice = bDevice;
    }

    void *symmPtr = aclshmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *symmetricPtr = reinterpret_cast<uint8_t *>(symmPtr);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling MM_RS kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        if (isNeedPaddingA && isNeedPaddingB) {
            ShmemMatmulReduceScatterPadding<true, true>
                <<<BLOCK_NUM, nullptr, stream>>>(
                util_get_ffts_config(),
                aDevice, bDevice, dDevice, waDevice, wbDevice, symmetricPtr,
                m, n, k
            );
        } else if (!isNeedPaddingA && isNeedPaddingB) {
            ShmemMatmulReduceScatterPadding<false, true>
                <<<BLOCK_NUM, nullptr, stream>>>(
                util_get_ffts_config(),
                aDevice, bDevice, dDevice, waDevice, wbDevice, symmetricPtr,
                m, n, k
            );
        } else if (isNeedPaddingA && !isNeedPaddingB) {
            ShmemMatmulReduceScatterPadding<true, false>
                <<<BLOCK_NUM, nullptr, stream>>>(
                util_get_ffts_config(),
                aDevice, bDevice, dDevice, waDevice, wbDevice, symmetricPtr,
                m, n, k
            );
        } else {
            ShmemMatmulReduceScatterPadding<false, false>
                <<<BLOCK_NUM, nullptr, stream>>>(
                util_get_ffts_config(),
                aDevice, bDevice, dDevice, waDevice, wbDevice, symmetricPtr,
                m, n, k
            );
        }
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling MM_RS kernel " << std::endl;

    ACL_CHECK(aclrtMemcpy(dHost, dSizeScatter, dDevice, dSizeScatter, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.GetDataPath("aclshmem_output.bin"), dHost, dSizeScatter, rank_id * dSizeScatter);
    if (rank_id == 0) {
        std::printf("test finished\n");
    }

    aclshmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(dHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(dDevice));
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtFree(waDevice));
    }
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(wbDevice));
    }

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
