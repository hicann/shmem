/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "host_device/shmemi_rdma_cqe_layout.h"
#include "utils/exception/shmem_exception_rdma_dump_internal.h"

namespace {
namespace rdma = aclshmemi::exception::rdma;

static_assert(sizeof(aclshmemi_rdma_exception_report_raw_t) == 272U, "RDMA raw snapshot ABI changed");
static_assert(
    offsetof(aclshmemi_rdma_exception_report_entry_t, cqe) < offsetof(aclshmemi_rdma_exception_report_entry_t, wq),
    "RDMA exception entry must keep the structured CQE before WQ");
static_assert(
    offsetof(aclshmemi_rdma_exception_report_entry_t, wq) < offsetof(aclshmemi_rdma_exception_report_entry_t, cq),
    "RDMA exception entry must keep WQ before CQ");
static_assert(
    offsetof(aclshmemi_rdma_exception_report_entry_t, cq) < offsetof(aclshmemi_rdma_exception_report_entry_t, raw),
    "RDMA exception entry must keep bounded raw bytes last");

TEST(RdmaExceptionDumpTest, ValidatesCqMetadata)
{
    EXPECT_TRUE(rdma::IsValidCqMetadata(0x1000U, 0x2000U, 8U, 32U));
    EXPECT_FALSE(rdma::IsValidCqMetadata(0U, 0x2000U, 8U, 32U));
    EXPECT_FALSE(rdma::IsValidCqMetadata(0x1000U, 0U, 8U, 32U));
    EXPECT_FALSE(rdma::IsValidCqMetadata(0x1000U, 0x2000U, 0U, 32U));
    EXPECT_FALSE(rdma::IsValidCqMetadata(0x1000U, 0x2000U, 8U, 16U));
    EXPECT_FALSE(rdma::IsValidCqMetadata(0x1000U, 0x2000U, 8U, 257U));
#if !defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    EXPECT_FALSE(rdma::IsValidCqMetadata(0x1000U, 0x2000U, 7U, 32U));
#endif
}

TEST(RdmaExceptionDumpTest, WrapsTailZeroWithoutUnderflowingTheRing)
{
    EXPECT_EQ(rdma::CqeIndex(std::numeric_limits<uint32_t>::max(), 8U), 7U);
    EXPECT_EQ(rdma::CqeIndex(0U, 8U), 0U);
    EXPECT_EQ(rdma::CqeIndex(8U, 8U), 0U);
    EXPECT_EQ(rdma::CqeIndex(1U, 0U), 0U);
}

TEST(RdmaExceptionDumpTest, ValidatesCqeOwnerForCurrentBackend)
{
    aclshmemi_rdma_exception_report_cqe_t cqe{};
    cqe.valid_fields = ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID;
    // First lap, second lap, and uint32_t rollover, with both owner values.
    for (uint32_t index : {3U, 32771U, UINT32_MAX, 0U}) {
        for (uint32_t owner : {0U, 1U}) {
            cqe.owner = owner;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
            const bool device_ready = owner == ((index / 32768U) & 1U);
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
            const bool device_ready = (static_cast<uint32_t>((index & 32768U) == 0U) ^ owner) != 0U;
#else
            const bool device_ready = (owner != 0U) != ((index & 32768U) != 0U);
#endif
            EXPECT_EQ(rdma::IsCqeReady(cqe, index, 32768U), device_ready);
        }
    }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    cqe.valid_fields |= ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID;
    cqe.owner = 0U;
    cqe.opcode = 0x1fU;
    EXPECT_FALSE(rdma::IsCqeReady(cqe, 3U, 32768U));
#endif
    EXPECT_FALSE(rdma::IsCqeReady(cqe, 3U, 0U));
}

TEST(RdmaExceptionDumpTest, FindsLaterFaultWithoutPublishingConsumerTail)
{
    for (uint32_t tail : {0U, 7U, UINT32_MAX}) {
        const uint32_t original_tail = tail;
        std::vector<uint32_t> visited;
        uint32_t fault_index = 0U;
        rdma::VisitPendingCqes(tail, 8U, [&](uint32_t logical_index) {
            aclshmemi_rdma_exception_report_cqe_t cqe{};
            cqe.valid_fields =
                ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
            cqe.owner = (logical_index & 8U) != 0U;
#else
            cqe.owner = (logical_index & 8U) == 0U;
#endif
            EXPECT_TRUE(rdma::IsCqeReady(cqe, logical_index, 8U));
            visited.push_back(rdma::CqeIndex(logical_index, 8U));
            cqe.status = visited.size() == 2U ? 5U : 0U;
            if (cqe.status != 0U) {
                fault_index = logical_index;
                return false;
            }
            return true;
        });
        EXPECT_EQ(visited, (std::vector<uint32_t>{tail & 7U, (tail + 1U) & 7U}));
        EXPECT_EQ(fault_index, static_cast<uint32_t>(tail + 1U));
        EXPECT_EQ(tail, original_tail);
    }
}

TEST(RdmaExceptionDumpTest, PendingCqeScanStopsAtGapAndIsBoundedByRing)
{
    uint32_t count = 0U;
    rdma::VisitPendingCqes(UINT32_MAX, 8U, [&](uint32_t) {
        ++count;
        return false; // Failed read or unready entry: never skip a gap.
    });
    EXPECT_EQ(count, 1U);
    count = 0U;
    rdma::VisitPendingCqes(UINT32_MAX, 8U, [&](uint32_t) {
        ++count;
        return true;
    });
    EXPECT_EQ(count, 8U);
    count = 0U;
    rdma::VisitPendingCqes(0U, 0U, [&](uint32_t) {
        ++count;
        return true;
    });
    EXPECT_EQ(count, 0U);
}

TEST(RdmaExceptionDumpTest, DetectsBackendErrorIndependentlyOfZeroSyndrome)
{
    aclshmemi_rdma_exception_report_cqe_t cqe{};
    EXPECT_FALSE(rdma::IsErrorCqe(cqe));
    cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID;
    cqe.opcode = 0x1eU;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    EXPECT_TRUE(rdma::IsErrorCqe(cqe));
    uint32_t visits = 0;
    rdma::VisitPendingCqes(0U, 8U, [&](uint32_t) {
        ++visits;
        return !rdma::IsErrorCqe(cqe);
    });
    EXPECT_EQ(visits, 1U);
    cqe.opcode = 1U;
    cqe.status = 5U;
    EXPECT_FALSE(rdma::IsErrorCqe(cqe));
#else
    EXPECT_FALSE(rdma::IsErrorCqe(cqe));
    cqe.status = 5U;
    EXPECT_TRUE(rdma::IsErrorCqe(cqe));
#endif
}

TEST(RdmaExceptionDumpTest, OverrunOwnerUsesWqePhaseInsteadOfConsumerTail)
{
    aclshmemi_rdma_exception_report_cqe_t cqe{};
    cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID;
    cqe.wqe_id = 9U << 3U;
    cqe.owner = 1U;
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    EXPECT_FALSE(rdma::IsCqeReady(cqe, 0U, 8U));
    EXPECT_TRUE(rdma::IsOverrunCqeReady(cqe, 8U));
    cqe.owner = 0U;
    EXPECT_FALSE(rdma::IsOverrunCqeReady(cqe, 8U));
    cqe.owner = 1U;
    cqe.valid_fields &= ~ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID;
#endif
    EXPECT_FALSE(rdma::IsOverrunCqeReady(cqe, 8U));
    EXPECT_FALSE(rdma::IsOverrunCqeReady(cqe, 0U));
}

TEST(RdmaExceptionDumpTest, RejectsEntryAddressOverflow)
{
    uint64_t address = 0U;
    EXPECT_TRUE(rdma::CheckedEntryAddress(0x1000U, 3U, 64U, address));
    EXPECT_EQ(address, 0x10c0U);
    EXPECT_FALSE(rdma::CheckedEntryAddress(0U, 3U, 64U, address));
    EXPECT_FALSE(rdma::CheckedEntryAddress(0x1000U, 3U, 0U, address));
    EXPECT_FALSE(rdma::CheckedEntryAddress(std::numeric_limits<uint64_t>::max() - 31U, 1U, 64U, address));
    EXPECT_FALSE(rdma::CheckedEntryAddress(1U, std::numeric_limits<uint64_t>::max(), 2U, address));
}

TEST(RdmaExceptionDumpTest, ValidatesMrRangeAndRkey)
{
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0x100, 7, 0x1010, 0x20, 7), rdma::MrCheckResult::IN_MR);
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0x100, 7, 0x0ff0, 0x20, 7), rdma::MrCheckResult::ADDR_OUT_OF_MR);
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0x100, 7, 0x10f0, 0x20, 7), rdma::MrCheckResult::ADDR_OUT_OF_MR);
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0x100, 7, 0x1010, 0x20, 8), rdma::MrCheckResult::RKEY_MISMATCH);
}

TEST(RdmaExceptionDumpTest, RejectsInvalidMrRangeInputs)
{
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0, 7, 0x1010, 1, 7), rdma::MrCheckResult::INVALID_INPUT);
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0x100, 7, 0, 1, 7), rdma::MrCheckResult::INVALID_INPUT);
    EXPECT_EQ(rdma::CheckMrRange(0x1000, 0x100, 7, UINT64_MAX - 1, 4, 7), rdma::MrCheckResult::INVALID_INPUT);
}

TEST(RdmaExceptionDumpTest, FormatsOnlyValidStructuredCqeFields)
{
    aclshmemi_rdma_exception_report_cqe_t cqe{};
    cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_SYNDROME_VALID;
    cqe.owner = 1U;
    cqe.status = 7U;
    cqe.wqn = 5U;
    cqe.opcode = 9U;
    cqe.syndrome = 11U;
    EXPECT_EQ(rdma::FormatCqe(cqe), "owner=1 status=7 wqn=5 syndrome=11");
}

TEST(RdmaExceptionDumpTest, DecodesBackendCqeFixture)
{
    uint8_t raw[32]{};
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    aclshmemi_xscdv_diamond_cqe_v2 fixture{};
    fixture.owner = 1U;
    fixture.error_code = 7U;
    fixture.qp_id = 9U;
    fixture.msg_opcode = 11U;
    fixture.wqe_id = 13U;
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    aclshmemi_hns_1825_cqe_t fixture{};
    fixture.owner_id_qpn = (1U << 31U) | 9U;
    fixture.op_sr_wqebb = 0x1eU << 27U;
    fixture.syndrome = 7U;
    fixture.wqe_counter = 13U;
#else
    aclshmemi_cqe_ctx fixture{};
    fixture.byte4 = (1U << 7U) | (7U << 8U);
    fixture.byte16 = 9U;
#endif
    std::memcpy(raw, &fixture, sizeof(fixture));

    aclshmemi_rdma_exception_report_cqe_t decoded{};
    ASSERT_TRUE(rdma::DecodeRawCqe(raw, sizeof(raw), decoded));
    EXPECT_EQ(decoded.owner, 1U);
    // Verify readiness after decoding the same raw CQE across the phase boundary.
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE) || defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    EXPECT_FALSE(rdma::IsCqeReady(decoded, 3U, 32768U));
    EXPECT_TRUE(rdma::IsCqeReady(decoded, 32771U, 32768U));
#else
    EXPECT_TRUE(rdma::IsCqeReady(decoded, 3U, 32768U));
    EXPECT_FALSE(rdma::IsCqeReady(decoded, 32771U, 32768U));
#endif
    EXPECT_EQ(decoded.status, 7U);
    EXPECT_EQ(decoded.wqn, 9U);
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    EXPECT_EQ(decoded.opcode, 11U);
    EXPECT_EQ(decoded.wqe_id, 13U);
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    EXPECT_EQ(decoded.opcode, 0x1eU);
    EXPECT_EQ(decoded.syndrome, 7U);
    EXPECT_EQ(decoded.wqe_id, 13U);
#endif
}

TEST(RdmaExceptionDumpTest, RejectsShortCqeFixture)
{
    uint8_t raw[31]{};
    aclshmemi_rdma_exception_report_cqe_t decoded{};
    EXPECT_FALSE(rdma::DecodeRawCqe(nullptr, 32U, decoded));
    EXPECT_FALSE(rdma::DecodeRawCqe(raw, sizeof(raw), decoded));
}

} // namespace
