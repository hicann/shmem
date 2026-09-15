/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_DEVICE_RDMA_EXCEPTION_REPORT_KERNEL_H
#define ACLSHMEMI_DEVICE_RDMA_EXCEPTION_REPORT_KERNEL_H

#include <cstdint>

#include "acl/acl_rt.h"

constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_SUCCESS = 0xAC10U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_INVALID_PARAM = 0xAC11U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_MAX_RAW_SIZE = 256U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW = 0U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_WQ = 1U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQ = 2U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQE_RAW = 3U;

constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID = 1U << 0U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID = 1U << 1U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID = 1U << 2U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID = 1U << 3U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_SYNDROME_VALID = 1U << 4U;
constexpr uint32_t ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID = 1U << 5U;

struct aclshmemi_rdma_exception_report_cqe_t {
    uint32_t valid_fields;
    uint32_t owner;
    uint32_t status;
    uint32_t wqn;
    uint32_t opcode;
    uint32_t syndrome;
    uint32_t wqe_id;
};

struct aclshmemi_rdma_exception_report_wq_t {
    uint32_t wqn;
    uint64_t buf_addr;
    uint32_t wqe_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    int32_t db_mode;
    uint64_t db_addr;
    uint32_t sl;
    uint64_t amo_addr;
    uint32_t amo_lkey;
    uint64_t db_sw_addr;
    uint8_t mtu_shift;
    uint8_t db_cos;
};

struct aclshmemi_rdma_exception_report_cq_t {
    uint32_t cqn;
    uint64_t buf_addr;
    uint32_t cqe_size;
    uint32_t depth;
    uint64_t head_addr;
    uint64_t tail_addr;
    int32_t db_mode;
    uint64_t db_addr;
    uint32_t cq_attr_flags;
    uint64_t db_sw_addr;
};

struct aclshmemi_rdma_exception_report_raw_t {
    uint64_t addr;
    uint32_t size;
    uint8_t data[ACLSHMEMI_RDMA_EXCEPTION_REPORT_MAX_RAW_SIZE];
};

struct aclshmemi_rdma_exception_report_entry_t {
    uint32_t ret;
    uint32_t entry_type;
    aclshmemi_rdma_exception_report_cqe_t cqe;
    aclshmemi_rdma_exception_report_wq_t wq;
    aclshmemi_rdma_exception_report_cq_t cq;
    aclshmemi_rdma_exception_report_raw_t raw;
};

static_assert(sizeof(aclshmemi_rdma_exception_report_raw_t) == 272U, "RDMA raw report ABI changed.");
static_assert(sizeof(aclshmemi_rdma_exception_report_cqe_t) == 28U, "RDMA CQE report ABI changed.");
static_assert(sizeof(aclshmemi_rdma_exception_report_entry_t) <= 512U, "RDMA report entry exceeds bounded size.");

int32_t aclshmemi_rdma_exception_report_read_entry_on_stream(
    uint32_t entry_type, uint64_t addr, uint64_t size, aclshmemi_rdma_exception_report_entry_t* out,
    aclrtStream stream = nullptr);

#endif // ACLSHMEMI_DEVICE_RDMA_EXCEPTION_REPORT_KERNEL_H
