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
#include <runtime/rt_ffts.h>
#include <fstream>
#include <string>
#include <algorithm>

#include <iostream>
#include <vector>
#include <cstring>
#include <sstream>
#include <map>
#include <ctime>
#include <iomanip>

#include "host/shmem_host_def.h"
#include "host/mem/shmem_host_heap.h"
#include "host/init/shmem_host_init.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/team/shmem_host_team.h"

#include "utils.h"
#include "info.h"
#include "tiling.h"
#include "launch_map.h"
#include "coc_tiling_lut.h"

using half = __fp16;

const std::map<CocCommType, std::string> commTypeMap = {
    { MATMUL_ALLREDUCE, "MatmulAllReduce" },
    { ALLGATHER_MATMUL, "AllGatherMatmul" },
    { MATMUL_REDUCE_SCATTER, "MatmulReduceScatter" },
    { MATMUL_REDUCE_SCATTER_PADDING, "MatmulReduceScatterPadding" },
    { MATMUL_REDUCE_SCATTER_PADDING_AB, "MatmulReduceScatterPaddingAB" },
    { MATMUL_REDUCE_SCATTER_PADDING_A, "MatmulReduceScatterPaddingA" },
    { MATMUL_REDUCE_SCATTER_PADDING_B, "MatmulReduceScatterPaddingB" },
    { ALLGATHER_MATMUL_WITH_GATHER_RESULT, "AllGatherMatmulWithGatherResult" },
    { ALLGATHER_MATMUL_PADDING, "AllGatherMatmulPadding" },
};

const uint32_t COMM_TILE_M = 64;
const uint32_t COMM_INTERVAL = 3;
const uint32_t COMM_NPU_SPLIT = 1;
const uint32_t COMM_DATA_SPLIT = 16;
const uint32_t COMM_BLOCK_M = 64;

struct Options {
    CocCommType commType;
    CocDataType dataType;
    int rankSize;
    int rankId;
    std::string ipPort{};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::vector<int> deviceIdList{};
    uint32_t test_start_line{0};
    uint32_t test_collect_rows{0};
    std::string parentPath{};
    std::string csv_file{};
    std::string data_file{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            COMM_TYPE_INDEX = 1,
            DATA_TYPE_INDEX,
            RANK_SIZE_INDEX,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            START_LINE_INDEX,
            COLLECT_ROWS_INDEX,
            PARENT_PATH_INDEX,
            CSV_FILE_INDEX,
            DEVICE_LIST_INDEX,
            DATA_FILE_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            return -1;
        }

        commType = static_cast<CocCommType>(std::atoi(argv[COMM_TYPE_INDEX]));
        dataType = static_cast<CocDataType>(std::atoi(argv[DATA_TYPE_INDEX]));
        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        test_start_line = std::atoi(argv[START_LINE_INDEX]);
        test_collect_rows = std::atoi(argv[COLLECT_ROWS_INDEX]);
        parentPath = argv[PARENT_PATH_INDEX];
        csv_file = argv[CSV_FILE_INDEX];
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
        if (argc > DATA_FILE_INDEX) {
            data_file = argv[DATA_FILE_INDEX];
        }
        return 0;
    }
};

std::vector<std::vector<uint32_t>> InitTestShapes(const Options &options)
{
    uint32_t startLine = options.test_start_line;
    uint32_t collectRows = options.test_collect_rows;
    std::string shapeFileName = options.csv_file;
    std::vector<std::string> headers = {};
    std::vector<std::vector<uint32_t>> shapes = {};
    std::ifstream file(shapeFileName);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << shapeFileName << std::endl;
        return shapes;
    }

    std::string line;

    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        while (getline(ss, header, ',')) {
            headers.push_back(header);
        }
    } else {
        std::cerr << "The file is empty or the header line fails to be read." << std::endl;
        return shapes;
    }

    int rowIndex = 0;
    int added = 0;

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (rowIndex < startLine) {
            ++rowIndex;
            continue;
        }
        if (added >= collectRows) {
            break;
        }

        std::stringstream ss(line);
        std::vector<uint32_t> shape;
        std::string cell;
        while (getline(ss, cell, ',')) {
            shape.push_back(std::stoi(cell));
        }

        if (shape.size() != headers.size()) {
            std::cerr << "The number of data columns in row " << rowIndex <<
                " does not match the number of header columns: " << line << std::endl;
        } else {
            shapes.push_back(shape);
            ++added;
        }
        ++rowIndex;
    }
    file.close();

    return shapes;
}

std::string GetCurrentTime()
{
    std::time_t now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);

    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return ss.str();
}
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

    return ACLSHMEM_SUCCESS;
}
int main(int argc, char **argv)
{
    int status = ACLSHMEM_SUCCESS;
    Options options;
    options.Parse(argc, argv);
    CocCommType commType = options.commType;
    CocDataType dataType = options.dataType;
    int n_ranks = options.rankSize;
    int rank_id = options.rankId;
    std::string ipPort = options.ipPort;
    int32_t deviceId = options.deviceIdList[rank_id];
    std::string data_file = options.data_file;
    const std::vector<std::vector<uint32_t>> shapes = InitTestShapes(options);

    std::cout << "[TEST] input rank_size: " << n_ranks << " rank_id: " << rank_id << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    // status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_init_attr_t attributes;
    test_set_attr(rank_id, n_ranks, local_mem_size, ipPort.c_str(), &attributes);
    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    std::string currentTime = GetCurrentTime();
    std::string currentDir = options.parentPath;
    std::string tilingFileName = currentDir + "/output/tiling/tilingData_" + currentTime + ".csv";
    if (rank_id == 0) {
        CreateTilingFile(tilingFileName);
    }

    for (size_t i = 0; i < shapes.size(); i++) {
        uint32_t m = shapes[i][0];
        uint32_t k = shapes[i][1];
        uint32_t n = shapes[i][2];
        uint32_t transA = shapes[i][3];
        uint32_t transB = shapes[i][4];
        CocCommType kernelType = commType;

        CocTilingParams cocTiling;
        cocTiling.m = m;
        cocTiling.n = n;
        cocTiling.k = k;
        COCMatMulInfo info{ int64_t(m), int64_t(k), int64_t(n) };
        cocTiling.m0 = M0;
        cocTiling.n0 = N0;
        cocTiling.k0 = K0;
        cocTiling.commTileM = COMM_TILE_M;
        cocTiling.commInterval = COMM_INTERVAL;
        cocTiling.commNpuSplit = COMM_NPU_SPLIT;
        cocTiling.commDataSplit = COMM_DATA_SPLIT;
        cocTiling.commBlockM = COMM_BLOCK_M;
        cocTiling.rankSize = n_ranks;

        size_t aSize = static_cast<size_t>(m) * k * sizeof(half);
        size_t bSize = static_cast<size_t>(k) * n * sizeof(half);
        size_t cSize = static_cast<size_t>(m) * n * sizeof(half);
        size_t cSizePerRank;
        size_t gatherASize = aSize * n_ranks;
        size_t wASize = 0;
        size_t wBSize = 0;
        if (commType == MATMUL_REDUCE_SCATTER) {
            cSizePerRank = cSize / n_ranks;
        } else if (commType == MATMUL_REDUCE_SCATTER_PADDING) {
            cSizePerRank = cSize / n_ranks;

            bool isNeedPaddingA = IsNeedPadding(m, k, transA);
            bool isNeedPaddingB = IsNeedPadding(k, n, transB);
            if (isNeedPaddingA && isNeedPaddingB) {
                kernelType = MATMUL_REDUCE_SCATTER_PADDING_AB;
                wASize = GetWorkspaceLen(m, k, M0, K0) * sizeof(half);
                wBSize = GetWorkspaceLen(k, n, K0, N0) * sizeof(half);
            } else if (isNeedPaddingA && !isNeedPaddingB) {
                kernelType = MATMUL_REDUCE_SCATTER_PADDING_A;
                wASize = GetWorkspaceLen(m, k, M0, K0) * sizeof(half);
            } else if (!isNeedPaddingA && isNeedPaddingB) {
                kernelType = MATMUL_REDUCE_SCATTER_PADDING_B;
                wBSize = GetWorkspaceLen(k, n, K0, N0) * sizeof(half);
            } else {
                kernelType = MATMUL_REDUCE_SCATTER;
            }
        } else if (commType == ALLGATHER_MATMUL || commType == ALLGATHER_MATMUL_WITH_GATHER_RESULT) {
            cSizePerRank = cSize * n_ranks;
        } else if (commType == ALLGATHER_MATMUL_PADDING) {
            cSizePerRank = cSize * n_ranks;

            bool isNeedPaddingB = IsNeedPadding(k, n, transB);
            if (isNeedPaddingB) {
                kernelType = ALLGATHER_MATMUL_PADDING;
                wBSize = GetWorkspaceLen(k, n, K0, N0) * sizeof(half);
            } else {
                kernelType = ALLGATHER_MATMUL;
            }
        } else {
            cSizePerRank = cSize;
        }

        std::string opName = commTypeMap.at(kernelType);

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (data_file != "") {
            ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize));
            ReadFile(data_file + "/rank_" + std::to_string(rank_id) + "_a.bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixA(m * k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bHost;
        if (data_file != "") {
            ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize));
            ReadFile(data_file + "/rank_" + std::to_string(rank_id) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(k * n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&cDevice), cSizePerRank, ACL_MEM_MALLOC_HUGE_FIRST));
        if (commType == MATMUL_REDUCE_SCATTER || commType == MATMUL_REDUCE_SCATTER_PADDING) {
            std::vector<uint8_t> matrixCInit(cSizePerRank, 0);
            ACL_CHECK(aclrtMemcpy(cDevice, cSizePerRank, matrixCInit.data(), cSizePerRank, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *wADevice{nullptr};
        if (wASize != 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&wADevice), wASize, ACL_MEM_MALLOC_HUGE_FIRST));
        } else {
            wADevice = aDevice;
        }

        uint8_t *wBDevice{nullptr};
        if (wBSize != 0) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&wBDevice), wBSize, ACL_MEM_MALLOC_HUGE_FIRST));
        } else {
            wBDevice = bDevice;
        }

        uint8_t *gatherADevice{nullptr};
        if (commType == ALLGATHER_MATMUL_WITH_GATHER_RESULT) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&gatherADevice), gatherASize, ACL_MEM_MALLOC_HUGE_FIRST));
            wADevice = gatherADevice;
        }

        void *symmPtr = aclshmem_malloc(ACLSHMEM_BUFF_BYTES);
        uint8_t *gmSymmetric = (uint8_t *)symmPtr;

        uint32_t warmUpTimes = std::getenv("WARM_UP_TIMES") == nullptr ? WARM_UP_TIMES :
                               std::stoull(std::getenv("WARM_UP_TIMES"));
        uint32_t perfTestCycleTimes = std::getenv("PERF_TEST_CYCLE_TIMES") == nullptr ? PERF_TEST_CYCLE_TIMES :
                                      std::stoull(std::getenv("PERF_TEST_CYCLE_TIMES"));
        uint32_t searchparams = (std::getenv("SEARCH_PARAMS") == nullptr) ? 1U :
                                 std::stoul(std::getenv("SEARCH_PARAMS"));

        std::vector<CocTilingParams> cocTilings;
        if (warmUpTimes == 0) {
            cocTilings.push_back(cocTiling);
        } else {
            if (searchparams == 1) {
                // 搜索 tiling
                GetTilings(cocTilings, cocTiling, commType, n_ranks);
            } else {
                bool ok = ApplyLookupTable(info, commType, n_ranks, cocTiling);
                if (!ok) {
                    std::cerr << "[LUT] no table for (" << opName << "," << n_ranks << "), using defaults\n";
                }
                cocTilings.push_back(cocTiling);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        auto kernelFunc = KernelDispatcher::GetKernelFunc(kernelType, dataType);

        for (size_t i = 0; i < warmUpTimes; i++) {
            kernelFunc(stream, fftsAddr, aDevice, bDevice, cDevice,
                       wADevice, wBDevice, gmSymmetric, cocTilings[0], transA, transB);
        }

        for (CocTilingParams tiling : cocTilings) {
            for (size_t i = 0; i < perfTestCycleTimes; i++) {
                kernelFunc(stream, fftsAddr, aDevice, bDevice,
                           cDevice, wADevice, wBDevice, gmSymmetric, tiling, transA, transB);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&cHost), cSizePerRank));
        ACL_CHECK(aclrtMemcpy(cHost, cSizePerRank, cDevice, cSizePerRank, ACL_MEMCPY_DEVICE_TO_HOST));

        uint8_t *gatherAHost;
        if (commType == ALLGATHER_MATMUL_WITH_GATHER_RESULT) {
            ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&gatherAHost), gatherASize));
            ACL_CHECK(aclrtMemcpy(gatherAHost, gatherASize, gatherADevice, gatherASize, ACL_MEMCPY_DEVICE_TO_HOST));
        }

        if (data_file != "") {
            if (commType == MATMUL_ALLREDUCE) {
                if (rank_id == 0) {
                    WriteFile(data_file + "/output.bin", cHost, cSizePerRank);
                }
            } else if (commType == ALLGATHER_MATMUL || commType == ALLGATHER_MATMUL_PADDING
                       || commType == ALLGATHER_MATMUL_WITH_GATHER_RESULT) {
                if (rank_id == 0) {
                    WriteFile(data_file + "/output.bin", cHost, cSizePerRank);
                    if (commType == ALLGATHER_MATMUL_WITH_GATHER_RESULT) {
                        WriteFile(data_file + "/output_gather_a.bin", gatherAHost, gatherASize);
                    }
                }
            } else if (commType == MATMUL_REDUCE_SCATTER || commType == MATMUL_REDUCE_SCATTER_PADDING) {
                WriteFile(data_file + "/output.bin", cHost, cSizePerRank, rank_id * cSizePerRank);
            }
        }

        if (rank_id == 0) {
            WriteTilingInfos(opName, cocTilings, tilingFileName, transA, transB);
            std::printf("M: %d, K: %d, N: %d aclrtSynchronizeStream success!\n", cocTiling.m, cocTiling.k, cocTiling.n);
        }

        aclshmem_free(symmPtr);

        if (data_file != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
        }
        if (wASize != 0) {
            ACL_CHECK(aclrtFree(wADevice));
        }
        if (wBSize != 0) {
            ACL_CHECK(aclrtFree(wBDevice));
        }
        ACL_CHECK(aclrtFreeHost(cHost));
        ACL_CHECK(aclrtFree(aDevice));
        ACL_CHECK(aclrtFree(bDevice));
        ACL_CHECK(aclrtFree(cDevice));
    }

    status = aclrtDestroyStream(stream);

    status = aclshmem_finalize();
    status = aclrtResetDevice(deviceId);
    status = aclFinalize();
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    return 0;
}