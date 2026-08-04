/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "acl/acl.h"
#include "opdev/fp16_t.h"
#include "shmem.h"
#include "tp_allreduce_udma_kernel.h"
#include "tp_allreduce_perf_host.h"
#include "utils.h"

using fp16_t = op::fp16_t;

namespace {

constexpr int TP_SIZE = 2;
constexpr int BOX_SIZE = 4;
constexpr uint32_t TAILCUT_PATH_COUNT = 3;
constexpr uint64_t LOCAL_MEM_SIZE = 1024UL * 1024UL * 1024UL;

template <typename T>
bool ParseIntegerArg(std::string_view text, T& result)
{
    if (text.empty() || text.front() < '0' || text.front() > '9') {
        return false;
    }
    std::string value(text);
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || errno == ERANGE ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        return false;
    }
    result = static_cast<T>(parsed);
    return true;
}

struct Options {
    int nPes{0};
    int peId{0};
    std::string ipPort;
    int firstNpu{0};
    uint32_t elements{0};
    uint32_t iterations{0};
    uint32_t warmupIterations{0};
    uint32_t aivCount{0};
    std::string dataType;
    int perfMode{0};
    std::string perfCsvPath;
    TpAllReduceUdmaMode mode{TpAllReduceUdmaMode::BASELINE};
    uint32_t directWeight{2};
    uint32_t relay0Weight{1};
    uint32_t relay1Weight{1};

    static bool IsSupportedDataType(const std::string& value)
    {
        return value == "int" || value == "int32_t" || value == "float16_t";
    }

    static bool ParseMode(const std::string& value, TpAllReduceUdmaMode& parsedMode)
    {
        if (value == "baseline") {
            parsedMode = TpAllReduceUdmaMode::BASELINE;
            return true;
        }
        if (value == "tailcut") {
            parsedMode = TpAllReduceUdmaMode::TAILCUT;
            return true;
        }
        return false;
    }

    static bool ParseRatio(
        std::string_view value, uint32_t& parsedDirectWeight, uint32_t& parsedRelay0Weight,
        uint32_t& parsedRelay1Weight)
    {
        size_t firstSeparator = value.find(':');
        if (firstSeparator == std::string_view::npos) {
            return false;
        }
        size_t secondSeparator = value.find(':', firstSeparator + 1);
        if (secondSeparator == std::string_view::npos ||
            value.find(':', secondSeparator + 1) != std::string_view::npos) {
            return false;
        }
        return ParseIntegerArg(value.substr(0, firstSeparator), parsedDirectWeight) &&
               ParseIntegerArg(
                   value.substr(firstSeparator + 1, secondSeparator - firstSeparator - 1), parsedRelay0Weight) &&
               ParseIntegerArg(value.substr(secondSeparator + 1), parsedRelay1Weight);
    }

    bool Parse(int argc, char** argv)
    {
        if (argc != 14) {
            return false;
        }
        int argIdx = 1;
        if (!ParseIntegerArg(argv[argIdx++], nPes) || !ParseIntegerArg(argv[argIdx++], peId)) {
            return false;
        }
        ipPort = argv[argIdx++];
        if (!ParseIntegerArg(argv[argIdx++], firstNpu) || !ParseIntegerArg(argv[argIdx++], elements) ||
            !ParseIntegerArg(argv[argIdx++], iterations) || !ParseIntegerArg(argv[argIdx++], aivCount)) {
            return false;
        }
        dataType = argv[argIdx++];
        if (!ParseIntegerArg(argv[argIdx++], perfMode)) {
            return false;
        }
        perfCsvPath = argv[argIdx++];
        if (!ParseIntegerArg(argv[argIdx++], warmupIterations) || !ParseMode(argv[argIdx++], mode) ||
            !ParseRatio(argv[argIdx++], directWeight, relay0Weight, relay1Weight)) {
            return false;
        }
        return argIdx == argc;
    }

    bool Validate() const
    {
        bool commonValid = nPes > 0 && peId >= 0 && peId < nPes && !ipPort.empty() && firstNpu >= 0 &&
                           nPes % TP_SIZE == 0 && elements > 0 && elements % TP_SIZE == 0 && iterations > 0 &&
                           aivCount > 0 && IsSupportedDataType(dataType) && (perfMode == 0 || perfMode == 1) &&
                           (perfMode != 0 || (iterations == 1 && warmupIterations == 0));
        if (!commonValid) {
            return false;
        }
        if (mode == TpAllReduceUdmaMode::BASELINE) {
            return directWeight > 0 && relay0Weight == 0 && relay1Weight == 0;
        }
        return nPes % BOX_SIZE == 0 && aivCount >= TAILCUT_PATH_COUNT &&
               (directWeight > 0 || relay0Weight > 0 || relay1Weight > 0);
    }

    bool IsModeAvailable() const
    {
#if defined(ACLSHMEM_RELAY_SUPPORT)
        return true;
#else
        return mode == TpAllReduceUdmaMode::BASELINE;
#endif
    }

    const char* ModeName() const { return mode == TpAllReduceUdmaMode::BASELINE ? "baseline" : "tailcut"; }

    std::string PerfOpName() const
    {
        if (mode == TpAllReduceUdmaMode::BASELINE) {
            return "tp_allreduce_udma_baseline";
        }
        return "tp_allreduce_udma_tailcut_ratio_" + std::to_string(directWeight) + "_" + std::to_string(relay0Weight) +
               "_" + std::to_string(relay1Weight);
    }

    std::string GetRankDataPath(const std::string& fileName) const
    {
        return "golden/shape_" + std::to_string(elements) + "_" + std::to_string(nPes) + "_" + std::to_string(TP_SIZE) +
               "_" + dataType + "/rank_" + std::to_string(peId) + "/" + fileName;
    }

    std::string GetOutputPath() const { return "output/output_" + std::to_string(peId) + ".bin"; }
};

aclshmemx_uniqueid_t defaultFlagUid;

int InitRuntime(const Options& options, int32_t& deviceId, aclrtStream& stream)
{
    deviceId = options.peId + options.firstNpu;
    int status = aclInit(nullptr);
    if (status != ACLSHMEM_SUCCESS) {
        return status;
    }
    status = aclrtSetDevice(deviceId);
    if (status != ACLSHMEM_SUCCESS) {
        return status;
    }
    status = aclrtCreateStream(&stream);
    if (status != ACLSHMEM_SUCCESS) {
        return status;
    }

    aclshmemx_init_attr_t attributes{};
    status =
        test_set_attr(options.peId, options.nPes, LOCAL_MEM_SIZE, options.ipPort.c_str(), defaultFlagUid, &attributes);
    if (status != ACLSHMEM_SUCCESS) {
        return status;
    }
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
}

void FinalizeRuntime(int32_t deviceId, aclrtStream stream)
{
    aclshmem_finalize();
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

template <typename T>
void LaunchTpAllReduceUdma(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, TpAllReduceUdmaMode mode, TpAllReduceUdmaStage stage, uint32_t directWeight,
    uint32_t relay0Weight, uint32_t relay1Weight);

template <>
void LaunchTpAllReduceUdma<int32_t>(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, TpAllReduceUdmaMode mode, TpAllReduceUdmaStage stage, uint32_t directWeight,
    uint32_t relay0Weight, uint32_t relay1Weight)
{
    launch_tp_allreduce_udma_int32_t(
        blockDim, stream, fftsConfig, workspace, elements, tpTeam, perfMode, static_cast<int32_t>(mode),
        static_cast<int32_t>(stage), directWeight, relay0Weight, relay1Weight);
}

template <>
void LaunchTpAllReduceUdma<fp16_t>(
    uint32_t blockDim, void* stream, uint64_t fftsConfig, uint8_t* workspace, uint32_t elements, aclshmem_team_t tpTeam,
    int32_t perfMode, TpAllReduceUdmaMode mode, TpAllReduceUdmaStage stage, uint32_t directWeight,
    uint32_t relay0Weight, uint32_t relay1Weight)
{
    launch_tp_allreduce_udma_float16_t(
        blockDim, stream, fftsConfig, workspace, elements, tpTeam, perfMode, static_cast<int32_t>(mode),
        static_cast<int32_t>(stage), directWeight, relay0Weight, relay1Weight);
}

template <typename T>
void LaunchTpAllReduceIteration(
    const Options& options, aclrtStream stream, uint64_t fftsConfig, uint8_t* workspace, aclshmem_team_t tpTeam,
    int32_t perfMode)
{
    LaunchTpAllReduceUdma<T>(
        options.aivCount, stream, fftsConfig, workspace, options.elements, tpTeam, perfMode, options.mode,
        TpAllReduceUdmaStage::REDUCE_SCATTER, options.directWeight, options.relay0Weight, options.relay1Weight);
    aclshmemx_barrier_on_stream(tpTeam, stream);
    LaunchTpAllReduceUdma<T>(
        options.aivCount, stream, fftsConfig, workspace, options.elements, tpTeam, perfMode, options.mode,
        TpAllReduceUdmaStage::LOCAL_REDUCE_ALLGATHER, options.directWeight, options.relay0Weight, options.relay1Weight);
    aclshmemx_barrier_on_stream(tpTeam, stream);
}

template <typename T>
int RunTpAllReduceCase(const Options& options, int32_t deviceId, aclrtStream stream, aclshmem_team_t tpTeam)
{
    const size_t tensorBytes = static_cast<size_t>(options.elements) * sizeof(T);
    const size_t workspaceElements = static_cast<size_t>(options.elements) * 3;
    const size_t workspaceBytes = workspaceElements * sizeof(T);
    uint8_t* workspace = static_cast<uint8_t*>(aclshmem_malloc(workspaceBytes));
    if (workspace == nullptr) {
        std::cerr << "[ERROR] symmetric workspace allocation failed, pe=" << options.peId << "\n";
        return 1;
    }

    int status = ACLSHMEM_SUCCESS;
    std::vector<T> initialData(workspaceElements, T{});
    if (!ReadFile(options.GetRankDataPath("input_gm.bin"), initialData.data(), tensorBytes)) {
        std::cerr << "[ERROR] failed to prepare input data, pe=" << options.peId << "\n";
        aclshmem_free(workspace);
        return 1;
    }
    status = aclrtMemcpy(workspace, workspaceBytes, initialData.data(), workspaceBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] failed to copy input to device, pe=" << options.peId << ", status=" << status << "\n";
        aclshmem_free(workspace);
        return status;
    }

    // Prevent a fast PE from remotely updating a peer workspace that the peer is still initializing.
    aclshmem_barrier(tpTeam);

    uint64_t fftsConfig = util_get_ffts_config();
    for (uint32_t i = 0; i < options.warmupIterations; ++i) {
        LaunchTpAllReduceIteration<T>(options, stream, fftsConfig, workspace, tpTeam, 0);
    }
    if (options.warmupIterations > 0) {
        status = aclrtSynchronizeStream(stream);
        if (status != ACLSHMEM_SUCCESS) {
            std::cerr << "[ERROR] warmup failed, pe=" << options.peId << ", status=" << status << "\n";
            aclshmem_free(workspace);
            return status;
        }
    }

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < options.iterations; ++i) {
        LaunchTpAllReduceIteration<T>(options, stream, fftsConfig, workspace, tpTeam, options.perfMode);
    }
    status = aclrtSynchronizeStream(stream);
    auto end = std::chrono::steady_clock::now();

    double elapsedUs = std::chrono::duration<double, std::micro>(end - start).count();
    double averageUs = elapsedUs / options.iterations;

    std::vector<T> output(options.elements);
    const size_t outputOffsetBytes = tensorBytes * 2;
    if (status == ACLSHMEM_SUCCESS) {
        status = aclrtMemcpy(
            output.data(), tensorBytes, workspace + outputOffsetBytes, tensorBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (status == ACLSHMEM_SUCCESS && !WriteFile(options.GetOutputPath(), output.data(), tensorBytes)) {
        std::cerr << "[ERROR] failed to write output data, pe=" << options.peId << "\n";
        status = 1;
    }

    if (status == ACLSHMEM_SUCCESS) {
        int rankInTp = options.peId % TP_SIZE;
        int peerGlobalRank = options.peId / TP_SIZE * TP_SIZE + (rankInTp ^ 1);
        std::cout << "[SUCCESS] pe=" << options.peId << ", device_id=" << deviceId
                  << ", tp_group=" << options.peId / TP_SIZE << ", rank_in_tp=" << rankInTp
                  << ", peer=" << peerGlobalRank << ", dtype=" << options.dataType << ", elements=" << options.elements
                  << ", aiv_count=" << options.aivCount << ", mode=" << options.ModeName();
        if (options.mode == TpAllReduceUdmaMode::TAILCUT) {
            std::cout << ", tailcut_ratio=" << options.directWeight << ":" << options.relay0Weight << ":"
                      << options.relay1Weight;
        }
        std::cout << ", warmup=" << options.warmupIterations << ", iterations=" << options.iterations
                  << ", avg_us=" << std::fixed << std::setprecision(3) << averageUs << "\n";
        if (options.perfMode != 0) {
            tp_allreduce_perf::Report(
                {options.nPes, options.peId, TP_SIZE, options.elements, options.aivCount, options.iterations,
                 options.dataType, options.perfCsvPath},
                options.PerfOpName(), averageUs, options.aivCount);
        }
    } else {
        std::cerr << "[FAILED] pe=" << options.peId << ", status=" << status << "\n";
    }

    // Keep every PE's symmetric heap and teams alive until all independent TP groups finish.
    aclshmem_barrier_all();
    aclshmem_free(workspace);
    return status;
}

int main(int argc, char** argv)
{
    Options options;
    if (!options.Parse(argc, argv) || !options.Validate()) {
        std::cerr << "Usage: tp_allreduce_udma n_pes pe_id ipport first_npu elements iterations aiv_count "
                     "data_type perf_mode perf_csv warmup_iterations mode direct:relay0:relay1\n"
                  << "Constraints: tp_size=2, n_pes%2=0, elements%2=0, iterations>0, aiv_count>0, "
                     "mode=baseline|tailcut; tailcut requires n_pes%4=0 and aiv_count>=3\n";
        return 1;
    }
    if (!options.IsModeAvailable()) {
        std::cerr << "[ERROR] tailcut mode requires ACLSHMEM_RELAY_SUPPORT; rebuild with -enable_relay\n";
        return 1;
    }

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int status = InitRuntime(options, deviceId, stream);
    if (status != ACLSHMEM_SUCCESS) {
        std::cerr << "[ERROR] runtime initialization failed, pe=" << options.peId << ", status=" << status << "\n";
        return status;
    }

    aclshmem_team_t tpTeam = ACLSHMEM_TEAM_INVALID;
    aclshmem_team_t crossTpTeam = ACLSHMEM_TEAM_INVALID;
    status = aclshmem_team_split_2d(ACLSHMEM_TEAM_WORLD, TP_SIZE, &tpTeam, &crossTpTeam);
    if (status != ACLSHMEM_SUCCESS || tpTeam == ACLSHMEM_TEAM_INVALID) {
        std::cerr << "[ERROR] TP team creation failed, pe=" << options.peId << ", status=" << status << "\n";
        FinalizeRuntime(deviceId, stream);
        return status == ACLSHMEM_SUCCESS ? 1 : status;
    }

    if (options.dataType == "int" || options.dataType == "int32_t") {
        status = RunTpAllReduceCase<int32_t>(options, deviceId, stream, tpTeam);
    } else if (options.dataType == "float16_t") {
        status = RunTpAllReduceCase<fp16_t>(options, deviceId, stream, tpTeam);
    } else {
        std::cerr << "[ERROR] unsupported data type: " << options.dataType << "\n";
        status = 1;
    }

    aclshmem_team_destroy(tpTeam);
    if (crossTpTeam != ACLSHMEM_TEAM_INVALID) {
        aclshmem_team_destroy(crossTpTeam);
    }
    FinalizeRuntime(deviceId, stream);
    return status;
}
