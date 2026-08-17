/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TP_ALLREDUCE_PERF_HOST_H
#define TP_ALLREDUCE_PERF_HOST_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"

namespace tp_allreduce_perf {

constexpr int PROF_FRAME_TOTAL = 0;
constexpr int PROF_FRAME_REDUCE_SCATTER = 1;
constexpr int PROF_FRAME_LOCAL_REDUCE = 2;
constexpr int PROF_FRAME_ALLGATHER = 3;
constexpr int PROF_FRAME_COUNT = 4;

struct ReportOptions {
    int nPes;
    int peId;
    int tpSize;
    uint32_t elements;
    uint32_t aivCount;
    uint32_t iterations;
    std::string dataType;
    std::string perfCsvPath;
};

struct FrameSummary {
    double maxBlockUs{0.0};
    int64_t count{0};
    std::vector<double> blockUs;
};

inline const char* FrameName(int frameId)
{
    switch (frameId) {
        case PROF_FRAME_TOTAL:
            return "device_total";
        case PROF_FRAME_REDUCE_SCATTER:
            return "reduce_scatter";
        case PROF_FRAME_LOCAL_REDUCE:
            return "local_reduce";
        case PROF_FRAME_ALLGATHER:
            return "allgather";
        default:
            return "unknown";
    }
}

inline int GetCycleToUs()
{
    const char* socName = aclrtGetSocName();
    if (socName != nullptr && std::string(socName).find("Ascend950") != std::string::npos) {
        return 1000;
    }
    return 50;
}

inline int GetProfPe()
{
    const char* profPe = std::getenv("SHMEM_CYCLE_PROF_PE");
    if (profPe == nullptr) {
        return -1;
    }
    return std::atoi(profPe);
}

inline std::string JoinBlockTimes(const std::vector<double>& blockUs)
{
    std::string result;
    for (size_t i = 0; i < blockUs.size(); ++i) {
        if (i > 0) {
            result += "|";
        }
        result += double_to_string(blockUs[i]);
    }
    return result;
}

inline FrameSummary GetFrameSummary(aclshmem_prof_pe_t* profs, int frameId, uint32_t blockNum)
{
    FrameSummary summary;
    if (profs == nullptr || frameId < 0 || frameId >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
        return summary;
    }

    uint32_t actualBlocks = std::min(blockNum, static_cast<uint32_t>(ACLSHMEM_CYCLE_PROF_MAX_BLOCK));
    int cycleToUs = GetCycleToUs();
    for (uint32_t blockId = 0; blockId < actualBlocks; ++blockId) {
        aclshmem_prof_block_t* prof = &profs->block_prof[blockId];
        if (prof->ccount[frameId] == 0) {
            continue;
        }
        double avgUs = static_cast<double>(prof->cycles[frameId]) / static_cast<double>(prof->ccount[frameId]) /
                       static_cast<double>(cycleToUs);
        summary.maxBlockUs = std::max(summary.maxBlockUs, avgUs);
        summary.count = std::max(summary.count, prof->ccount[frameId]);
        summary.blockUs.push_back(avgUs);
    }
    return summary;
}

inline void WriteCsvRows(
    const ReportOptions& options, const std::string& opName, double hostAvgUs,
    const std::vector<FrameSummary>& summaries)
{
    if (options.perfCsvPath.empty()) {
        return;
    }
    std::string dir = get_dir(options.perfCsvPath);
    if (!dir.empty() && access(dir.c_str(), F_OK) != 0) {
        make_dir(dir);
    }

    std::ofstream out(options.perfCsvPath, std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[WARN] failed to open perf csv: " << options.perfCsvPath << "\n";
        return;
    }
    out << "case_id,op,pes,tp_size,pe,dtype,elements,aiv_count,iterations,metric,frame_id,"
           "max_block_us,count,block_us\n";

    std::string caseId = "shape_" + std::to_string(options.elements) + "_" + std::to_string(options.nPes) + "_" +
                         std::to_string(options.tpSize) + "_" + options.dataType;
    out << caseId << "," << opName << "," << options.nPes << "," << options.tpSize << "," << options.peId << ","
        << options.dataType << "," << options.elements << "," << options.aivCount << "," << options.iterations
        << ",host_total,-1," << double_to_string(hostAvgUs) << "," << options.iterations << ",\n";
    for (int frameId = 0; frameId < PROF_FRAME_COUNT; ++frameId) {
        const FrameSummary& summary = summaries[frameId];
        out << caseId << "," << opName << "," << options.nPes << "," << options.tpSize << "," << options.peId << ","
            << options.dataType << "," << options.elements << "," << options.aivCount << "," << options.iterations
            << "," << FrameName(frameId) << "," << frameId << "," << double_to_string(summary.maxBlockUs) << ","
            << summary.count << "," << JoinBlockTimes(summary.blockUs) << "\n";
    }
}

inline void Report(const ReportOptions& options, const std::string& opName, double hostAvgUs, uint32_t blockNum)
{
    int profPe = GetProfPe();
    if (profPe != options.peId) {
        return;
    }

    aclshmem_prof_pe_t* outProfs = nullptr;
    aclshmemx_get_prof(&outProfs, false);
    if (outProfs == nullptr) {
        std::cerr << "[WARN] profiling is enabled but no profile data was collected, pe=" << options.peId << "\n";
        return;
    }

    std::vector<FrameSummary> summaries(PROF_FRAME_COUNT);
    for (int frameId = 0; frameId < PROF_FRAME_COUNT; ++frameId) {
        summaries[frameId] = GetFrameSummary(outProfs, frameId, blockNum);
    }

    std::cout << "[PERF] pe=" << options.peId << ", metric=host_total, avg_us=" << double_to_string(hostAvgUs) << "\n";
    for (int frameId = 0; frameId < PROF_FRAME_COUNT; ++frameId) {
        const FrameSummary& summary = summaries[frameId];
        std::cout << "[PERF] pe=" << options.peId << ", metric=" << FrameName(frameId) << ", frame=" << frameId
                  << ", max_block_us=" << double_to_string(summary.maxBlockUs) << ", count=" << summary.count
                  << ", block_us=" << JoinBlockTimes(summary.blockUs) << "\n";
    }

    WriteCsvRows(options, opName, hostAvgUs, summaries);
}

} // namespace tp_allreduce_perf

#endif // TP_ALLREDUCE_PERF_HOST_H
