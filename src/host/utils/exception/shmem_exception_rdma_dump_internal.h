/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_EXCEPTION_RDMA_DUMP_INTERNAL_H
#define ACLSHMEMI_EXCEPTION_RDMA_DUMP_INTERNAL_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "utils/exception/shmemi_device_rdma_exception_report_kernel.h"

namespace aclshmemi::exception::rdma {

constexpr uint32_t kMinCqeSize = 32U;
constexpr uint32_t kMaxCqeSize = 256U;
constexpr uint32_t kMaxCqDepth = 1U << 20U;

bool IsValidCqMetadata(uint64_t buffer, uint64_t tail, uint32_t depth, uint32_t stride);
bool CheckedEntryAddress(uint64_t base, uint64_t index, uint64_t stride, uint64_t& address);
enum class MrCheckResult { IN_MR, ADDR_OUT_OF_MR, RKEY_MISMATCH, INVALID_INPUT };
MrCheckResult CheckMrRange(
    uint64_t mr_addr, uint64_t mr_size, uint32_t mr_rkey, uint64_t remote_addr, uint64_t remote_len,
    uint32_t remote_rkey);
// Visit an unpublished completion batch without changing the device consumer index.
// The callback returns false on read failure, an unready entry, or the fault CQE.
template <typename Visitor>
void VisitPendingCqes(uint32_t tail, uint32_t depth, Visitor visit)
{
    for (uint32_t offset = 0; offset < depth; ++offset) {
        if (!visit(static_cast<uint32_t>(tail + offset))) {
            break;
        }
    }
}

uint32_t CqeIndex(uint32_t logical_index, uint32_t depth);
bool IsCqeReady(const aclshmemi_rdma_exception_report_cqe_t& cqe, uint32_t logical_index, uint32_t depth);
bool IsErrorCqe(const aclshmemi_rdma_exception_report_cqe_t& cqe);
bool IsOverrunCqeReady(const aclshmemi_rdma_exception_report_cqe_t& cqe, uint32_t depth);
bool DecodeRawCqe(const uint8_t* raw, size_t size, aclshmemi_rdma_exception_report_cqe_t& cqe);
std::string FormatCqe(const aclshmemi_rdma_exception_report_cqe_t& cqe);

} // namespace aclshmemi::exception::rdma

#endif // ACLSHMEMI_EXCEPTION_RDMA_DUMP_INTERNAL_H
