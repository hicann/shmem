/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/exception/shmem_exception_rdma_dump_internal.h"

#include <cstring>
#include <algorithm>
#include <limits>
#include <sstream>

#include "host_device/shmemi_rdma_cqe_layout.h"

namespace aclshmemi::exception::rdma {

bool IsValidCqMetadata(uint64_t buffer, uint64_t tail, uint32_t depth, uint32_t stride)
{
    const bool power_of_two = depth != 0U && (depth & (depth - 1U)) == 0U;
    return buffer != 0U && tail != 0U && depth != 0U && depth <= kMaxCqDepth && stride >= kMinCqeSize &&
           stride <= kMaxCqeSize &&
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
           true;
#else
           power_of_two;
#endif
}

bool CheckedEntryAddress(uint64_t base, uint64_t index, uint64_t stride, uint64_t& address)
{
    if (base == 0U || stride == 0U || index > std::numeric_limits<uint64_t>::max() / stride) {
        return false;
    }
    const uint64_t offset = index * stride;
    if (offset > std::numeric_limits<uint64_t>::max() - base) {
        return false;
    }
    address = base + offset;
    return true;
}

MrCheckResult CheckMrRange(
    uint64_t mr_addr, uint64_t mr_size, uint32_t mr_rkey, uint64_t remote_addr, uint64_t remote_len,
    uint32_t remote_rkey)
{
    if (mr_size == 0U || remote_addr == 0U || remote_len > UINT64_MAX - remote_addr || mr_size > UINT64_MAX - mr_addr) {
        return MrCheckResult::INVALID_INPUT;
    }
    if (remote_addr < mr_addr || remote_addr + remote_len > mr_addr + mr_size) {
        return MrCheckResult::ADDR_OUT_OF_MR;
    }
    return remote_rkey == mr_rkey ? MrCheckResult::IN_MR : MrCheckResult::RKEY_MISMATCH;
}

uint32_t CqeIndex(uint32_t logical_index, uint32_t depth)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    return depth == 0U ? 0U : logical_index % depth;
#else
    return depth == 0U ? 0U : logical_index & (depth - 1U);
#endif
}

bool IsCqeReady(const aclshmemi_rdma_exception_report_cqe_t& cqe, uint32_t logical_index, uint32_t depth)
{
    if (depth == 0U || (cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID) == 0U) {
        return false;
    }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    const uint32_t expected_owner = (logical_index / depth) & 1U;
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    // Match the device check: owner differs from (logical_index & depth) == 0.
    const uint32_t expected_owner = ((logical_index & depth) == 0U) ? 0U : 1U;
    constexpr uint32_t kInvalidOpcode = 0x1fU;
    if ((cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID) != 0U && cqe.opcode == kInvalidOpcode) {
        return false;
    }
#else
    // IN_DIE waits while owner equals the consumer-index phase.
    const uint32_t expected_owner = ((logical_index & depth) == 0U) ? 1U : 0U;
#endif
    return cqe.owner == expected_owner;
}

bool IsErrorCqe(const aclshmemi_rdma_exception_report_cqe_t& cqe)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    return (cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID) != 0U && cqe.opcode == 0x1eU;
#else
    return (cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID) != 0U && cqe.status != 0U;
#endif
}

bool IsOverrunCqeReady(const aclshmemi_rdma_exception_report_cqe_t& cqe, uint32_t depth)
{
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    return (cqe.valid_fields & ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID) != 0U &&
           IsCqeReady(cqe, cqe.wqe_id >> 3U, depth);
#else
    return false;
#endif
}

bool DecodeRawCqe(const uint8_t* raw, size_t size, aclshmemi_rdma_exception_report_cqe_t& cqe)
{
    cqe = {};
    if (raw == nullptr || size < sizeof(aclshmemi_cqe_ctx)) {
        return false;
    }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID;
    if constexpr (ACLSHMEMI_XSCALE_API_VERSION_VAR == 1) {
        aclshmemi_xscdv_diamond_cqe_v1 raw_cqe{};
        std::copy_n(raw, sizeof(raw_cqe), reinterpret_cast<uint8_t*>(&raw_cqe));
        cqe.owner = raw_cqe.owner;
        cqe.status = raw_cqe.error_code;
        cqe.wqn = raw_cqe.qp_id;
        cqe.opcode = raw_cqe.msg_opcode;
        cqe.wqe_id = raw_cqe.wqe_id;
    } else {
        aclshmemi_xscdv_diamond_cqe_v2 raw_cqe{};
        std::copy_n(raw, sizeof(raw_cqe), reinterpret_cast<uint8_t*>(&raw_cqe));
        cqe.owner = raw_cqe.owner;
        cqe.status = raw_cqe.error_code;
        cqe.wqn = raw_cqe.qp_id;
        cqe.opcode = raw_cqe.msg_opcode;
        cqe.wqe_id = raw_cqe.wqe_id;
    }
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    aclshmemi_hns_1825_cqe_t raw_cqe{};
    std::copy_n(raw, sizeof(raw_cqe), reinterpret_cast<uint8_t*>(&raw_cqe));
    cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_SYNDROME_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID;
    cqe.owner = (raw_cqe.owner_id_qpn >> 31U) & 1U;
    cqe.wqn = raw_cqe.owner_id_qpn & 0xfffffU;
    cqe.opcode = (raw_cqe.op_sr_wqebb >> 27U) & 0x1fU;
    cqe.syndrome = raw_cqe.syndrome;
    cqe.status = cqe.opcode == 0x1eU ? cqe.syndrome : 0U;
    cqe.wqe_id = raw_cqe.wqe_counter;
#else
    aclshmemi_cqe_ctx raw_cqe{};
    std::copy_n(raw, sizeof(raw_cqe), reinterpret_cast<uint8_t*>(&raw_cqe));
    cqe.valid_fields = ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID |
                       ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID;
    cqe.owner = (raw_cqe.byte4 >> 7U) & 1U;
    cqe.status = (raw_cqe.byte4 >> 8U) & 0xffU;
    cqe.wqn = raw_cqe.byte16 & 0xffffffU;
#endif
    return true;
}

std::string FormatCqe(const aclshmemi_rdma_exception_report_cqe_t& cqe)
{
    std::ostringstream out;
    auto append = [&](uint32_t field, const char* name, uint32_t value) {
        if ((cqe.valid_fields & field) != 0U) {
            out << name << '=' << value << ' ';
        }
    };
    append(ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID, "owner", cqe.owner);
    append(ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID, "status", cqe.status);
    append(ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID, "wqn", cqe.wqn);
    append(ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID, "opcode", cqe.opcode);
    append(ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_SYNDROME_VALID, "syndrome", cqe.syndrome);
    append(ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID, "wqeId", cqe.wqe_id);
    std::string result = out.str();
    if (!result.empty()) {
        result.pop_back();
    }
    return result;
}

} // namespace aclshmemi::exception::rdma
