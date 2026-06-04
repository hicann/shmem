/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_RMA_PERFTEST_ARGPARSER
#define SIMT_RMA_PERFTEST_ARGPARSER

#include <iostream>
#include <string>
#include <optional>
#include <iomanip>
#include <cstdint>

enum class OpType { Get, Put, None };
enum class VfType { Simt, Simd };

inline OpType str2op(const std::string& s) {
    if (s == "get") {
        return OpType::Get;
    } else if (s == "put") {
        return OpType::Put;
    } else if (s == "none") {
        return OpType::None;
    } else {
        throw std::invalid_argument("Invalid operation type");
    }
}

inline VfType str2vf(const std::string& s) {
    if (s == "simt") {
        return VfType::Simt;
    } else if (s == "simd") {
        return VfType::Simd;
    } else {
        throw std::invalid_argument("Invalid vectorization type");
    }
}

inline std::string to_string(OpType op) {
    switch (op) {
        case OpType::Get:
            return "get";
        case OpType::Put:
            return "put";
        case OpType::None:
            return "none";
        default:
            throw std::invalid_argument("Invalid operation type");
    }
}

inline std::string to_string(VfType vf) {
    switch (vf) {
        case VfType::Simt:
            return "simt";
        case VfType::Simd:
            return "simd";
        default:
            throw std::invalid_argument("Invalid vectorization type");
    }
}

struct Config {
    int npes = 2;
    int mype = -1;
    int used_core = 32;
    int bytes_in_exp_min = 10;
    int bytes_in_exp_max = 14;
    int warmup_loops = 50;
    int loops = 1000;
    int ub_size = 16;

    friend std::ostream& operator<<(std::ostream& os, const Config& c) {
        os << "================ Config ================\n"
           << std::left << std::setw(20) << "npes:" << c.npes << "\n"
           << std::setw(20) << "mype:" << c.mype << "\n"
           << std::setw(20) << "used_core:" << c.used_core << "\n"
           << std::setw(20) << "bytes_in_exp_min:" << c.bytes_in_exp_min << "\n"
           << std::setw(20) << "bytes_in_exp_max:" << c.bytes_in_exp_max << "\n"
           << std::setw(20) << "warmup_loops:" << c.warmup_loops << "\n"
           << std::setw(20) << "loops:" << c.loops << "\n"
           << std::setw(20) << "ub_size:" << c.ub_size << "\n"
           << "========================================";
        return os;
    }
};

// ---------- 使用说明 ----------
inline void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --mype <int> [options]\n"
              << "Options:\n"
              << "  --mype <int>             Required\n"
              << "  --used_core <int>        Default: 32\n"
              << "  --bytes_in_exp_min <int> Default: 10\n"
              << "  --bytes_in_exp_max <int> Default: 14\n"
              << "  --warmup_loops <int>     Default: 50\n"
              << "  --loops <int>            Default: 1000\n"
              << "  --ub_size <int>          Default: 16\n"
              << "  --help                   Show this message\n";
}

// ---------- 参数校验 ----------
// 各项约束均来源于 main.cpp 中对 Config 字段的实际使用方式：
//  - mype  : 作为 aclrtSetDevice 的设备号，且固定两卡模型只区分
//            ACTIVE_PE(0) / PASSIVE_PE(1)，因此只能取 0 或 1。
//  - used_core : kernel 的 grid 大小，须为正；其上限（256KB * used_core
//            不超过对称堆容量）由调用方自行保证，这里不再校验。
//  - bytes_in_exp_* : per_invocation_bytes = 1 << exp，指数需落在
//            (4, 15) 即 [5, 14]，且 min <= max。
//  - warmup_loops 在 [0, 10000)；loops 在 [1, 10000)（loops 是求平均时的除数）。
//  - ub_size 为 UB 大小(KB)，须为正。
inline bool validate_config(const Config& c) {
    if (c.mype != 0 && c.mype != 1) {
        std::cerr << "Error: --mype must be 0 (ACTIVE_PE) or 1 (PASSIVE_PE), got "
                  << c.mype << ".\n";
        return false;
    }
    if (c.used_core < 1) {
        std::cerr << "Error: --used_core must be >= 1, got " << c.used_core << ".\n";
        return false;
    }
    if (c.bytes_in_exp_min <= 4 || c.bytes_in_exp_min >= 15) {
        std::cerr << "Error: --bytes_in_exp_min must be in (4, 15), got "
                  << c.bytes_in_exp_min << ".\n";
        return false;
    }
    if (c.bytes_in_exp_max <= 4 || c.bytes_in_exp_max >= 15) {
        std::cerr << "Error: --bytes_in_exp_max must be in (4, 15), got "
                  << c.bytes_in_exp_max << ".\n";
        return false;
    }
    if (c.bytes_in_exp_min > c.bytes_in_exp_max) {
        std::cerr << "Error: --bytes_in_exp_min (" << c.bytes_in_exp_min
                  << ") must not exceed --bytes_in_exp_max (" << c.bytes_in_exp_max << ").\n";
        return false;
    }
    if (c.warmup_loops < 0 || c.warmup_loops >= 10000) {
        std::cerr << "Error: --warmup_loops must be in [0, 10000), got "
                  << c.warmup_loops << ".\n";
        return false;
    }
    if (c.loops < 1 || c.loops >= 10000) {
        std::cerr << "Error: --loops must be in [1, 10000), got " << c.loops << ".\n";
        return false;
    }
    if (c.ub_size < 1) {
        std::cerr << "Error: --ub_size must be >= 1 (KB), got " << c.ub_size << ".\n";
        return false;
    }
    return true;
}

// ---------- 参数解析 ----------
std::optional<Config> parse_args(int argc, char* argv[]) {
    Config conf;
    bool mype_provided = false;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return std::nullopt;
            }

            if (i + 1 < argc) {
                i += 1;
                std::string val = argv[i];

                if (arg == "--mype") {
                    conf.mype = std::stoi(val);
                    mype_provided = true;
                } else if (arg == "--used_core") {
                    conf.used_core = std::stoi(val);
                } else if (arg == "--bytes_in_exp_min") {
                    conf.bytes_in_exp_min = std::stoi(val);
                } else if (arg == "--bytes_in_exp_max") {
                    conf.bytes_in_exp_max = std::stoi(val);
                } else if (arg == "--warmup_loops") {
                    conf.warmup_loops = std::stoi(val);
                } else if (arg == "--loops") {
                    conf.loops = std::stoi(val);
                } else if (arg == "--ub_size") {
                    conf.ub_size = std::stoi(val);
                } else {
                    std::cerr << "Unknown argument: " << arg << "\n";
                    return std::nullopt;
                }
            } else {
                std::cerr << "Argument " << arg << " missing value.\n";
                return std::nullopt;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return std::nullopt;
    }

    if (!mype_provided) {
        std::cerr << "Error: --mype is required.\n";
        return std::nullopt;
    }

    if (!validate_config(conf)) {
        return std::nullopt;
    }

    return conf;
}


template <int32_t data_size>
struct TraitTypeFromDataSize {  };

// Specializations for specific data sizes
template<>
struct TraitTypeFromDataSize<8> { using type = int8_t; };

template<>
struct TraitTypeFromDataSize<16> { using type = int16_t; };

template<>
struct TraitTypeFromDataSize<32> { using type = int32_t; };

template<>
struct TraitTypeFromDataSize<64> { using type = int64_t; };

#endif 