/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <limits>
#include <vector>
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"

int g_npus = 2;
const char* ipport = "tcp://127.0.0.1:8769";
int f_pe = 0;
int f_npu = 0;
aclshmemx_uniqueid_t default_flag_uid;
constexpr int SDMA_AIVS_PER_BLOCK = 2;

extern "C" void launch_sdma_perf_kernel(
    uint32_t block_dim, void* stream, uint64_t fftsAddr, uint8_t* dst_gva, uint8_t* src_gva, uint8_t* timing_out_gva,
    int elements, int32_t frame_id, int data_type, int64_t prof_pe, int loop_count, int ub_size_kb, bool bidirectional,
    bool is_put, uint32_t qp_num, int metric, int batch);

int validate_aiv_count(int device_id, const std::vector<int>& block_sizes)
{
    int64_t vector_core_num = 0;
    int status = aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_core_num);
    if (status != ACL_SUCCESS || vector_core_num <= 0) {
        status = aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vector_core_num);
    }
    if (status != ACL_SUCCESS || vector_core_num <= 0) {
        std::cerr << "Failed to query AIV count for device " << device_id << std::endl;
        return 1;
    }

    const int max_aiv_count = std::min<int64_t>(vector_core_num, ACLSHMEM_MAX_AIV_PER_NPU);
    for (int block_size : block_sizes) {
        if (block_size > max_aiv_count) {
            std::cerr << "SDMA AIV count " << block_size << " exceeds device " << device_id << " capacity "
                      << max_aiv_count << std::endl;
            return 1;
        }
    }
    return 0;
}

template <typename T>
int run_sdma_test(
    int pe_id, int n_pes, uint64_t local_mem_size, const std::vector<int>& block_sizes, int min_exponent,
    int max_exponent, int loop_count, perftest::perf_data_type_t data_type, int prof_pe, int ub_size_kb,
    bool bidirectional, bool is_put, uint32_t qp_num, perftest::perf_metric_t metric, int batch,
    std::vector<std::vector<std::string>>& csv)
{
    int device_id = pe_id % g_npus + f_npu;
    aclrtStream stream = nullptr;
    int status = aclInit(nullptr);
    if (status != ACL_SUCCESS || aclrtSetDevice(device_id) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "Failed to initialize ACL for PE " << pe_id << std::endl;
        return 1;
    }
    if (is_put) {
        const char* soc_name = aclrtGetSocName();
        if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
            std::cerr << "SDMA put is not supported on Ascend950" << std::endl;
            aclrtDestroyStream(stream);
            aclrtResetDevice(device_id);
            aclFinalize();
            return 1;
        }
    }
    if (validate_aiv_count(device_id, block_sizes) != 0) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    if (aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_SDMA, qp_num) != ACL_SUCCESS) {
        std::cerr << "Failed to configure " << qp_num << " SDMA QPs for PE " << pe_id << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    aclshmemx_init_attr_t attributes{};
    test_set_attr(pe_id, n_pes, local_mem_size, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_SDMA;
    if (aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes) != ACL_SUCCESS) {
        std::cerr << "Failed to initialize SDMA for PE " << pe_id << std::endl;
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    int result = 0;
    int frame_id = 0;
    for (int block_size : block_sizes) {
        for (int exponent = min_exponent; exponent <= max_exponent; ++exponent) {
            const size_t data_size = size_t{1} << exponent;
            if (data_size % sizeof(T) != 0) {
                std::cerr << "Data size must be a multiple of the data type size" << std::endl;
                result = 1;
                break;
            }
            const size_t total_elements = data_size / sizeof(T);
            // Each QP/AIV transfers a complete data_size message (see kernel), so reserve
            // one private region per active QP in each symmetric buffer.
            const size_t region_elements = total_elements * static_cast<size_t>(qp_num);
            const size_t region_bytes = data_size * static_cast<size_t>(qp_num);
            // Symmetric timing slots, rdma_perftest style: region 0 = PE0 per-AIV cycles,
            // region 1 = PE1 per-AIV cycles. Zeroed before each run so idle AIV slots read 0.
            constexpr size_t kTimingSlots = 2 * ACLSHMEM_MAX_AIV_PER_NPU;
            const size_t timing_bytes = kTimingSlots * sizeof(int64_t);
            void* dst = aclshmem_malloc(region_bytes);
            void* src = aclshmem_malloc(region_bytes);
            void* timing_out = aclshmem_malloc(timing_bytes);
            if (dst == nullptr || src == nullptr || timing_out == nullptr) {
                std::cerr << "Failed to allocate symmetric memory" << std::endl;
                if (dst != nullptr)
                    aclshmem_free(dst);
                if (src != nullptr)
                    aclshmem_free(src);
                if (timing_out != nullptr)
                    aclshmem_free(timing_out);
                result = 1;
                break;
            }
            std::vector<T> src_input(region_elements, static_cast<T>(pe_id + 10));
            std::vector<T> dst_input(region_elements, static_cast<T>(pe_id + 100));
            std::vector<int64_t> timing_zero(kTimingSlots, 0);
            if (aclrtMemcpy(src, region_bytes, src_input.data(), region_bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
                    ACL_SUCCESS ||
                aclrtMemcpy(dst, region_bytes, dst_input.data(), region_bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
                    ACL_SUCCESS ||
                aclrtMemcpy(timing_out, timing_bytes, timing_zero.data(), timing_bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
                    ACL_SUCCESS) {
                std::cerr << "Failed to initialize test buffers" << std::endl;
                aclshmem_free(dst);
                aclshmem_free(src);
                aclshmem_free(timing_out);
                result = 1;
                break;
            }

            std::cout << "pe: " << pe_id << " qps: " << qp_num << (is_put ? " put size: " : " get size: ") << data_size
                      << " frame_id: " << frame_id << std::endl;
            aclshmem_barrier_all();
            const uint32_t block_dim = (qp_num + SDMA_AIVS_PER_BLOCK - 1) / SDMA_AIVS_PER_BLOCK;
            const uint64_t fftsAddr = util_get_ffts_config();
            launch_sdma_perf_kernel(
                block_dim, stream, fftsAddr, static_cast<uint8_t*>(dst), static_cast<uint8_t*>(src),
                static_cast<uint8_t*>(timing_out), static_cast<int>(total_elements), frame_id,
                static_cast<int>(data_type), prof_pe, loop_count, ub_size_kb, bidirectional, is_put, qp_num,
                static_cast<int>(metric), batch);
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                std::cerr << "SDMA performance kernel failed" << std::endl;
                result = 1;
            }
            if (result == 0) {
                // Synchronize PEs once after all AIV SDMA requests have completed, matching examples/sdma.
                aclshmem_barrier_all();
            }

            std::vector<T> dst_output(region_elements);
            if (result == 0 &&
                aclrtMemcpy(dst_output.data(), region_bytes, dst, region_bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
                    ACL_SUCCESS) {
                std::cerr << "Failed to copy test result to host" << std::endl;
                result = 1;
            }
            const int put_target_pe = (prof_pe + 1) % n_pes;
            const bool should_verify = bidirectional || (is_put ? pe_id == put_target_pe : pe_id == prof_pe);
            if (result == 0 && should_verify) {
                const int source_pe = bidirectional ? (is_put ? (pe_id + n_pes - 1) % n_pes : (pe_id + 1) % n_pes) :
                                                      (is_put ? prof_pe : (prof_pe + 1) % n_pes);
                T expected = static_cast<T>(source_pe + 10);
                // Verify every per-AIV message region.
                size_t mismatch = 0;
                while (mismatch < region_elements && dst_output[mismatch] == expected)
                    ++mismatch;
                if (mismatch != region_elements) {
                    std::cerr << "[Verification] FAILED: dst[" << mismatch
                              << "]=" << static_cast<double>(dst_output[mismatch])
                              << ", expected=" << static_cast<double>(expected) << std::endl;
                    result = 1;
                } else {
                    std::cout << "[Verification] SUCCESS: "
                              << (is_put ? (bidirectional ? "bi_put" : "put") : (bidirectional ? "bi_get" : "get"))
                              << std::endl;
                }
            }

            if (result == 0) {
                std::vector<int64_t> timing_host(kTimingSlots, 0);
                if (aclrtMemcpy(
                        timing_host.data(), timing_bytes, timing_out, timing_bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
                    ACL_SUCCESS) {
                    std::cerr << "Failed to copy timing result to host" << std::endl;
                    result = 1;
                } else {
                    const char* soc_name = aclrtGetSocName();
                    int64_t cycle2us = 50;
                    if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
                        cycle2us = 1000;
                    }
                    const int active_aivs = std::min<uint32_t>(qp_num, ACLSHMEM_MAX_AIV_PER_NPU);
                    const int64_t* region0 = timing_host.data();
                    const int64_t* region1 = timing_host.data() + ACLSHMEM_MAX_AIV_PER_NPU;

                    // Per-AIV per-iteration time: worst case across both PEs (identical for
                    // symmetric runs, picks the active side for unilateral runs).
                    double core_max_us = 0.0;
                    for (int aiv = 0; aiv < active_aivs; ++aiv) {
                        const int64_t cycles = std::max(region0[aiv], region1[aiv]);
                        const double per_iter_us =
                            (loop_count > 0) ? static_cast<double>(cycles) / cycle2us / loop_count : 0.0;
                        core_max_us = std::max(core_max_us, per_iter_us);
                    }

                    // Per-PE window = slowest AIV on that PE.
                    int64_t pe0_cycles = 0;
                    int64_t pe1_cycles = 0;
                    for (int aiv = 0; aiv < active_aivs; ++aiv) {
                        pe0_cycles = std::max(pe0_cycles, region0[aiv]);
                        pe1_cycles = std::max(pe1_cycles, region1[aiv]);
                    }
                    const double per_iter_us0 =
                        (loop_count > 0) ? static_cast<double>(pe0_cycles) / cycle2us / loop_count : 0.0;
                    const double per_iter_us1 =
                        (loop_count > 0) ? static_cast<double>(pe1_cycles) / cycle2us / loop_count : 0.0;
                    // Every active QP carries one complete data_size message per iteration.
                    const double bytes_per_iter = static_cast<double>(data_size) * active_aivs;

                    double bw_gb = 0.0;
                    double bw_gib = 0.0;
                    double per_iter_us = 0.0;
                    if (bidirectional) {
                        if (metric == perftest::PERF_METRIC_LAT) {
                            per_iter_us = (per_iter_us0 + per_iter_us1) / 2;
                            std::cout << "[LAT] PE" << pe_id << " forward(PE0->PE1)=" << per_iter_us0 << " us"
                                      << " reverse(PE1->PE0)=" << per_iter_us1 << " us"
                                      << " avg=" << per_iter_us << " us" << std::endl;
                        } else {
                            const double bps0 = (per_iter_us0 > 0) ? bytes_per_iter / per_iter_us0 * 1000000.0 : 0.0;
                            const double bps1 = (per_iter_us1 > 0) ? bytes_per_iter / per_iter_us1 * 1000000.0 : 0.0;
                            const double bw_gb_0 = bps0 / 1000.0 / 1000.0 / 1000.0;
                            const double bw_gb_1 = bps1 / 1000.0 / 1000.0 / 1000.0;
                            bw_gb = bw_gb_0 + bw_gb_1;
                            bw_gib = (bps0 + bps1) / 1024.0 / 1024.0 / 1024.0;
                            per_iter_us = per_iter_us0 + per_iter_us1;
                            std::cout << "[BW] PE" << pe_id << " forward(PE0->PE1)=" << bw_gb_0 << " GB/s"
                                      << " reverse(PE1->PE0)=" << bw_gb_1 << " GB/s"
                                      << " total=" << bw_gb << " GB/s" << std::endl;
                        }
                    } else {
                        // Unilateral: only the runner PE (prof PE) has a nonzero window.
                        const double active_us = (prof_pe == 0) ? per_iter_us0 : per_iter_us1;
                        per_iter_us = active_us;
                        if (metric == perftest::PERF_METRIC_LAT) {
                            std::cout << "[LAT] PE" << pe_id << ": " << active_us << " us" << std::endl;
                        } else {
                            const double bps = (active_us > 0) ? bytes_per_iter / active_us * 1000000.0 : 0.0;
                            bw_gb = bps / 1000.0 / 1000.0 / 1000.0;
                            bw_gib = bps / 1024.0 / 1024.0 / 1024.0;
                            std::cout << "[BW] PE" << pe_id << ": " << bw_gb << " GB/s" << std::endl;
                        }
                    }

                    std::vector<std::string> row = {
                        uint64_to_string(data_size),   int_to_string(g_npus),   int_to_string(block_size),
                        int_to_string(ub_size_kb),     double_to_string(bw_gb), double_to_string(bw_gib),
                        double_to_string(core_max_us),
                    };
                    csv.push_back(row);
                }
            }
            aclshmem_free(dst);
            aclshmem_free(src);
            aclshmem_free(timing_out);
            if (result != 0 || ++frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT)
                break;
        }
        if (result != 0 || frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT)
            break;
    }
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return result;
}

int main(int argc, char* argv[])
{
    int n_pes = 2, pe_id = 0, min_exponent = 3, max_exponent = 17;
    int loop_count = 1000, ub_size_kb = 16, qp_num = 2;
    std::vector<int> block_sizes;
    const char* data_type_name = "float";
    const char* test_type = "get";
    const char* metric_str = "bw";
    int batch = 0; // 0 / unset => full async (== loop_count); 1 => sync; N => quiet every N submits
    static option options[] = {
        {"pes", required_argument, nullptr, 0},
        {"pe-id", required_argument, nullptr, 0},
        {"ipport", required_argument, nullptr, 0},
        {"gnpus", required_argument, nullptr, 0},
        {"fpe", required_argument, nullptr, 0},
        {"fnpu", required_argument, nullptr, 0},
        {"test-type", required_argument, nullptr, 't'},
        {"datatype", required_argument, nullptr, 'd'},
        {"qp", required_argument, nullptr, 0},
        {"exponent", required_argument, nullptr, 'e'},
        {"exponent-range", required_argument, nullptr, 0},
        {"loop-count", required_argument, nullptr, 0},
        {"ub-size", required_argument, nullptr, 0},
        {"metric", required_argument, nullptr, 0},
        {"batch", required_argument, nullptr, 0},
        {nullptr, 0, nullptr, 0}};
    int opt, index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:e:", options, &index)) != -1) {
        if (opt == 't')
            test_type = optarg;
        else if (opt == 'd')
            data_type_name = optarg;
        else if (opt == 'e')
            min_exponent = max_exponent = std::atoi(optarg);
        else if (opt == 0) {
            const char* name = options[index].name;
            if (std::strcmp(name, "pes") == 0)
                n_pes = std::atoi(optarg);
            else if (std::strcmp(name, "pe-id") == 0)
                pe_id = std::atoi(optarg);
            else if (std::strcmp(name, "ipport") == 0)
                ipport = optarg;
            else if (std::strcmp(name, "gnpus") == 0)
                g_npus = std::atoi(optarg);
            else if (std::strcmp(name, "fpe") == 0)
                f_pe = std::atoi(optarg);
            else if (std::strcmp(name, "fnpu") == 0)
                f_npu = std::atoi(optarg);
            else if (std::strcmp(name, "exponent-range") == 0 && optind < argc) {
                min_exponent = std::atoi(optarg);
                max_exponent = std::atoi(argv[optind++]);
            } else if (std::strcmp(name, "loop-count") == 0)
                loop_count = std::atoi(optarg);
            else if (std::strcmp(name, "ub-size") == 0)
                ub_size_kb = std::atoi(optarg);
            else if (std::strcmp(name, "qp") == 0)
                qp_num = std::atoi(optarg);
            else if (std::strcmp(name, "metric") == 0)
                metric_str = optarg;
            else if (std::strcmp(name, "batch") == 0)
                batch = std::atoi(optarg);
        } else
            return 1;
    }
    bool is_put = std::strcmp(test_type, "put") == 0 || std::strcmp(test_type, "bi_put") == 0;
    bool bidirectional = std::strcmp(test_type, "bi_get") == 0 || std::strcmp(test_type, "bi_put") == 0;
    if ((!bidirectional && !is_put && std::strcmp(test_type, "get") != 0) || n_pes < 2 || pe_id < 0 || pe_id >= n_pes ||
        g_npus < 1 || min_exponent < 0 || max_exponent < min_exponent ||
        max_exponent >= std::numeric_limits<int>::digits || loop_count < 1 || ub_size_kb < 1) {
        std::cerr << "Invalid arguments: supported types are get, bi_get, put and bi_put; require at least two PEs and "
                     "positive sizes"
                  << std::endl;
        return 1;
    }
    if (std::strcmp(metric_str, "bw") != 0 && std::strcmp(metric_str, "lat") != 0) {
        std::cerr << "Invalid arguments: --metric must be 'bw' or 'lat' (got '" << metric_str << "')" << std::endl;
        return 1;
    }
    perftest::perf_metric_t metric =
        std::strcmp(metric_str, "lat") == 0 ? perftest::PERF_METRIC_LAT : perftest::PERF_METRIC_BW;
    if (batch < 0) {
        std::cerr << "Invalid arguments: --batch must be >= 0 (got " << batch << ")" << std::endl;
        return 1;
    }
    if (batch == 0 || batch > loop_count) {
        batch = loop_count; // 0 or batch > loop_count => full async, single trailing quiet
    }
    if (qp_num < 1 || qp_num > ACLSHMEM_MAX_AIV_PER_NPU) {
        std::cerr << "SDMA qp count must be in [1, " << ACLSHMEM_MAX_AIV_PER_NPU << "]" << std::endl;
        return 1;
    }
    block_sizes = {qp_num};
    const char* prof_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    int prof_pe = prof_env == nullptr ? 0 : std::atoi(prof_env);
    if (prof_pe < 0 || prof_pe >= n_pes) {
        std::cerr << "SHMEM_CYCLE_PROF_PE is outside the PE range" << std::endl;
        return 1;
    }
    uint64_t local_mem_size = 1024ULL * 1024 * 1024;
    uint64_t max_size = uint64_t{1} << max_exponent;
    // Each QP carries a complete max_size message, so reserve one region per QP.
    const uint64_t per_pe_buffer_bytes = max_size * static_cast<uint64_t>(qp_num) * 2;
    if (per_pe_buffer_bytes > local_mem_size)
        local_mem_size = per_pe_buffer_bytes;
    std::vector<std::vector<std::string>> csv = {
        {"DataSize/B", "Npus", "QPs", "UBsize/KB", "Bandwidth/GB/s(1000)", "Bandwidth/GiB/s(1024)", "CoreMaxTime/us"}};
    perftest::perf_data_type_t data_type = get_data_type(data_type_name);
    int status = 0;
#define RUN_SDMA_TEST(type)                                                                                    \
    status = run_sdma_test<type>(                                                                              \
        pe_id, n_pes, local_mem_size, block_sizes, min_exponent, max_exponent, loop_count, data_type, prof_pe, \
        ub_size_kb, bidirectional, is_put, static_cast<uint32_t>(qp_num), metric, batch, csv)
    DISPATCH_BY_TYPE(data_type_name, RUN_SDMA_TEST);
#undef RUN_SDMA_TEST
    if (pe_id == prof_pe && status == 0) {
        // 文件名带 qp 数量后缀，区分不同 --qp 配置生成的结果文件。
        write_csv(
            "output/sdma_" + std::string(metric_str) + "_" + std::string(test_type) + "_" +
                std::string(data_type_name) + "_qp" + int_to_string(qp_num) + "_" + int_to_string(prof_pe) + ".csv",
            csv);
    }
    std::cout << (status == 0 ? "[SUCCESS]" : "[FAILED]") << " sdma_perftest done in pe " << pe_id << std::endl;
    return status;
}
