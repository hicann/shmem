/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"
#include "transpose_alltoall_device.h"

namespace {
constexpr uint64_t SHMEM_MALLOC_MAX_SIZE = 1024UL * 1024UL * 1024;
constexpr size_t SHMEM_BUFF_BYTES = 1004 * 1024 * 1024;

constexpr uint32_t AICORE_NUM = 28;
constexpr uint32_t WORKSPACE_STAGES = 4;
constexpr uint32_t TILE_M = 1024 * 1024;

constexpr int64_t IPC_BUFF_HALF_SIZE = (1024UL * 1024 * 1000) / 2;

constexpr uint32_t KERNEL_ITERATIONS = 5;

struct Options {
    static constexpr auto HELPER =
        "Usage: transpose_alltoall rank_size rank_id ip_port B S N D dtype data_path [device_id_list]\n"
        "  dtype: float16_t | float32_t | int32_t\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t B{0};
    uint32_t S{0};
    uint32_t N{0};
    uint32_t D{0};
    std::string dataType{"float16_t"};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char** argv)
    {
        enum class ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            B_INDEX,
            S_INDEX,
            N_INDEX,
            D_INDEX,
            DTYPE_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        int positionalArgc = argc - 1;
        if (positionalArgc < static_cast<int>(ArgsIndex::DATA_PATH_INDEX) ||
            positionalArgc > static_cast<int>(ArgsIndex::INDEX_MAX)) {
            printf("%s", HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_SIZE_INDEX)]);
        rankId = std::atoi(argv[static_cast<int>(ArgsIndex::RANK_ID_INDEX)]);
        if (rankSize <= 0) {
            ERROR_LOG("rank_size must be a positive integer");
            return -1;
        }
        ipPort = argv[static_cast<int>(ArgsIndex::IP_PORT_INDEX)];
        B = std::atoi(argv[static_cast<int>(ArgsIndex::B_INDEX)]);
        S = std::atoi(argv[static_cast<int>(ArgsIndex::S_INDEX)]);
        N = std::atoi(argv[static_cast<int>(ArgsIndex::N_INDEX)]);
        D = std::atoi(argv[static_cast<int>(ArgsIndex::D_INDEX)]);
        dataType = argv[static_cast<int>(ArgsIndex::DTYPE_INDEX)];
        dataPath = argv[static_cast<int>(ArgsIndex::DATA_PATH_INDEX)];
        if (argc > static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)) {
            char* idListStr = argv[static_cast<int>(ArgsIndex::DEVICE_LIST_INDEX)];
            for (char* idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < static_cast<size_t>(rankSize); ++i) {
                deviceIdList.push_back(static_cast<int>(i));
            }
        }
        if (deviceIdList.size() != static_cast<size_t>(rankSize)) {
            ERROR_LOG("device_id_list size (%zu) must equal rank_size (%d)", deviceIdList.size(), rankSize);
            return -1;
        }
        return 0;
    }

    std::string GetDataPath() const { return dataPath; }
};

TransposeAllToAllTiling SetupTiling(uint32_t B, uint32_t S, uint32_t N, uint32_t D, uint32_t rankSize)
{
    TransposeAllToAllTiling tiling{};
    tiling.B = B;
    tiling.S = S;
    tiling.N = N;
    tiling.D = D;
    tiling.rankSize = rankSize;
    return tiling;
}

bool CheckTiling(uint32_t rankSize, const TransposeAllToAllTiling& tiling, size_t elemBytes)
{
    if (tiling.B % rankSize != 0)
        return false;

    if (tiling.D <= 0)
        return false;
    if (tiling.S <= 0)
        return false;
    uint64_t srcStrideBytes = static_cast<uint64_t>(tiling.S) * tiling.D * elemBytes;
    if (srcStrideBytes > UINT32_MAX)
        return false;

    uint64_t dataSize = static_cast<uint64_t>(tiling.B) * tiling.S * tiling.N * tiling.D;
    uint64_t chunkSize = dataSize / rankSize;
    uint32_t D = tiling.D;

    uint32_t commSizeM = static_cast<uint32_t>((TILE_M < chunkSize) ? TILE_M : chunkSize);
    commSizeM = (commSizeM / D) * D;
    if (commSizeM == 0)
        commSizeM = D;

    uint64_t blocksPerChunk = (chunkSize + commSizeM - 1) / commSizeM;
    uint64_t totalBlocks = blocksPerChunk * rankSize;
    if (totalBlocks > static_cast<uint64_t>(INT32_MAX))
        return false;

    int64_t workspaceBytes =
        static_cast<int64_t>(WORKSPACE_STAGES) * AICORE_NUM * commSizeM * static_cast<int64_t>(elemBytes);
    return workspaceBytes <= IPC_BUFF_HALF_SIZE;
}

bool CheckAcl(aclError error, const char* opName)
{
    if (error != ACL_ERROR_NONE) {
        ERROR_LOG("%s failed, aclError=%d", opName, static_cast<int>(error));
        return false;
    }
    return true;
}

bool CheckNotNull(const void* ptr, const char* name)
{
    if (ptr == nullptr) {
        ERROR_LOG("%s returned nullptr", name);
        return false;
    }
    return true;
}

uint8_t* AllocateAndLoadInput(size_t bufferSize, const std::string& dataFile, const std::string& fileSuffix)
{
    uint8_t* devicePtr = nullptr;
    if (!CheckAcl(aclrtMalloc((void**)(&devicePtr), bufferSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)") ||
        !CheckNotNull(devicePtr, "aclrtMalloc(input)")) {
        return nullptr;
    }
    if (!dataFile.empty()) {
        uint8_t* hostPtr = nullptr;
        if (!CheckAcl(aclrtMallocHost((void**)(&hostPtr), bufferSize), "aclrtMallocHost(input)") ||
            !CheckNotNull(hostPtr, "aclrtMallocHost(input)")) {
            ACL_CHECK(aclrtFree(devicePtr));
            return nullptr;
        }
        bool ok = ReadFile(dataFile + fileSuffix, hostPtr, bufferSize);
        if (!ok) {
            ERROR_LOG("Read input file failed: %s", (dataFile + fileSuffix).c_str());
        } else {
            ok = CheckAcl(
                aclrtMemcpy(devicePtr, bufferSize, hostPtr, bufferSize, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(input H2D)");
        }
        ACL_CHECK(aclrtFreeHost(hostPtr));
        if (!ok) {
            ACL_CHECK(aclrtFree(devicePtr));
            return nullptr;
        }
    } else {
        std::vector<uint8_t> defaultData(bufferSize, 0);
        if (!CheckAcl(
                aclrtMemcpy(devicePtr, bufferSize, defaultData.data(), bufferSize, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(default input H2D)")) {
            ACL_CHECK(aclrtFree(devicePtr));
            return nullptr;
        }
    }
    return devicePtr;
}

bool WriteOutputFile(
    const std::string& dataFile, const std::string& fileSuffix, uint8_t* outputDevice, size_t outputSize)
{
    uint8_t* outputHost = nullptr;
    if (!CheckAcl(aclrtMallocHost((void**)(&outputHost), outputSize), "aclrtMallocHost(output)") ||
        !CheckNotNull(outputHost, "aclrtMallocHost(output)")) {
        return false;
    }
    bool ok = CheckAcl(
        aclrtMemcpy(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy(output D2H)");
    if (ok) {
        ok = WriteFile(dataFile + fileSuffix, outputHost, outputSize);
    }
    ACL_CHECK(aclrtFreeHost(outputHost));
    return ok;
}

} // namespace

template <typename Element>
int RunTransposeAllToAll(const Options& options)
{
    const int rankSize = options.rankSize;
    const int rankId = options.rankId;

    TransposeAllToAllTiling tiling =
        SetupTiling(options.B, options.S, options.N, options.D, static_cast<uint32_t>(rankSize));
    if (!CheckTiling(static_cast<uint32_t>(rankSize), tiling, sizeof(Element))) {
        ERROR_LOG("Invalid tiling!");
        return -1;
    }

    aclrtStream stream = nullptr;
    uint8_t* inputPtr = nullptr;
    uint8_t* outputPtr = nullptr;
    void* symmPtr = nullptr;
    int32_t* prepStatusPtr = nullptr; // 对称区：各 rank 的准备状态，calloc 零初始化

    auto cleanup = [&]() {
        if (prepStatusPtr != nullptr) {
            aclshmem_free(prepStatusPtr);
            prepStatusPtr = nullptr;
        }
        if (symmPtr != nullptr) {
            aclshmem_free(symmPtr);
            symmPtr = nullptr;
        }
        if (inputPtr != nullptr) {
            ACL_CHECK(aclrtFree(inputPtr));
            inputPtr = nullptr;
        }
        if (outputPtr != nullptr) {
            ACL_CHECK(aclrtFree(outputPtr));
            outputPtr = nullptr;
        }
        if (stream != nullptr) {
            ACL_CHECK(aclrtDestroyStream(stream));
            stream = nullptr;
        }
    };
    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << "  B=" << options.B
              << "  S=" << options.S << "  N=" << options.N << "  D=" << options.D << "  dtype=" << options.dataType
              << " (" << sizeof(Element) << "B)" << std::endl;

    const int64_t inputSize =
        static_cast<int64_t>(options.B) * options.S * options.N * options.D * static_cast<int64_t>(sizeof(Element));
    const size_t ioBytes = static_cast<size_t>(inputSize);

    symmPtr = aclshmem_calloc(1, SHMEM_BUFF_BYTES);
    prepStatusPtr = static_cast<int32_t*>(aclshmem_calloc(static_cast<size_t>(rankSize), sizeof(int32_t)));
    bool localOk =
        CheckNotNull(symmPtr, "aclshmem_calloc(symm)") && CheckNotNull(prepStatusPtr, "aclshmem_calloc(prepStatus)");
    if (!localOk) {
        if (symmPtr != nullptr) {
            aclshmem_free(symmPtr);
            symmPtr = nullptr;
        }
        if (prepStatusPtr != nullptr) {
            aclshmem_free(prepStatusPtr);
            prepStatusPtr = nullptr;
        }
    }

    if (localOk) {
        localOk =
            CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") && CheckNotNull(stream, "aclrtCreateStream");
    }
    if (localOk) {
        inputPtr =
            AllocateAndLoadInput(ioBytes, options.GetDataPath(), "/rank_" + std::to_string(rankId) + "_input.bin");
        localOk = CheckNotNull(inputPtr, "AllocateAndLoadInput");
    }
    if (localOk) {
        localOk =
            CheckAcl(aclrtMalloc((void**)(&outputPtr), ioBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(outputPtr)") &&
            CheckNotNull(outputPtr, "aclrtMalloc(outputPtr)");
    }
    if (localOk) {
        localOk = CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(before kernel)");
    }

    if (prepStatusPtr != nullptr) {
        const int32_t myStatus = localOk ? 1 : 0;
        for (int peer = 0; peer < rankSize; ++peer) {
            aclshmem_int32_p(prepStatusPtr + rankId, myStatus, peer);
        }
    }
    aclshmem_barrier_all();

    bool allOk = localOk;
    if (localOk) {
        std::vector<int32_t> peerStatus(static_cast<size_t>(rankSize), 0);
        if (!CheckAcl(
                aclrtMemcpy(
                    peerStatus.data(), peerStatus.size() * sizeof(int32_t), prepStatusPtr,
                    peerStatus.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(prepStatus D2H)")) {
            allOk = false;
        } else {
            for (int peer = 0; peer < rankSize; ++peer) {
                if (peerStatus[static_cast<size_t>(peer)] != 1) {
                    ERROR_LOG("rankId %d preparation not ready (status=%d)", peer, peerStatus[peer]);
                    allOk = false;
                }
            }
        }
    }
    if (!allOk) {
        ERROR_LOG("Preparation agreement failed, rankId=%d localOk=%d, all ranks skip kernel", rankId, localOk ? 1 : 0);
        cleanup();
        return -1;
    }

    uint8_t* symmetricPtr = static_cast<uint8_t*>(symmPtr);
    const uint32_t curBlockNum = AICORE_NUM;
    const uint64_t fftsAddr = shmemx_get_ffts_config();

    std::cout << "Before calling Ascend950TransposeAllToAll kernel" << std::endl;
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL); // 清空历史错误，保证发射错误检测准确
    for (uint32_t i = 0; i < KERNEL_ITERATIONS; i++) {
        Ascend950TransposeAllToAll<Element, Element>
            <<<curBlockNum, nullptr, stream>>>(fftsAddr, inputPtr, outputPtr, symmetricPtr, tiling);
    }
    if (!CheckAcl(aclrtGetLastError(ACL_RT_THREAD_LEVEL), "Ascend950TransposeAllToAll launch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(after kernel)")) {
        cleanup();
        return -1;
    }
    std::cout << "After calling Ascend950TransposeAllToAll kernel" << std::endl;

    if (!WriteOutputFile(
            options.GetDataPath(), "/rank_" + std::to_string(rankId) + "_output.bin", outputPtr, ioBytes)) {
        ERROR_LOG("Write output file failed, rankId=%d", rankId);
        cleanup();
        return -1;
    }
    std::printf("rankId %d test finished\n", rankId);

    cleanup();
    return 0;
}

int main(int argc, char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    if (options.rankId < 0 || options.rankId >= options.rankSize) {
        ERROR_LOG("Invalid rankId");
        return 1;
    }
    const int32_t deviceId = options.deviceIdList[static_cast<size_t>(options.rankId)];

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));

    aclshmemx_init_attr_t attributes{};
    aclshmemx_uniqueid_t default_flag_uid{};
    test_set_attr(
        options.rankId, options.rankSize, SHMEM_MALLOC_MAX_SIZE, options.ipPort.c_str(), default_flag_uid, &attributes);
    int initStatus = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (initStatus != ACLSHMEM_SUCCESS) {
        ERROR_LOG("aclshmemx_init_attr failed, rankId=%d, status=%d", options.rankId, initStatus);
        ACL_CHECK(aclrtResetDevice(deviceId));
        ACL_CHECK(aclFinalize());
        return 1;
    }

    int status;
    const std::string& dataType = options.dataType;
    if (dataType == "float16_t") {
        status = RunTransposeAllToAll<half>(options);
    } else if (dataType == "float32_t" || dataType == "float") {
        status = RunTransposeAllToAll<float>(options);
    } else if (dataType == "int32_t" || dataType == "int") {
        status = RunTransposeAllToAll<int32_t>(options);
    } else {
        ERROR_LOG("Unsupported data type: %s (expected float16_t/float32_t/int32_t)", dataType.c_str());
        status = 1;
    }

    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    std::cout << "[TEST] exit rankId: " << options.rankId << std::endl;
    return status;
}
