/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_RDMA_CQE_LAYOUT_H
#define ACLSHMEMI_RDMA_CQE_LAYOUT_H

#include <cstdint>

constexpr int ACLSHMEMI_XSCALE_API_VERSION_VAR = 2;
static_assert(
    ACLSHMEMI_XSCALE_API_VERSION_VAR == 1 || ACLSHMEMI_XSCALE_API_VERSION_VAR == 2,
    "ACLSHMEMI_XSCALE_API_VERSION must be 1 or 2");

struct aclshmemi_xscdv_diamond_cqe_v1 {
    uint32_t error_code : 8;
    uint32_t qp_id : 15;
    uint32_t rsv : 1;
    uint32_t se : 1;
    uint32_t has_pph : 1;
    uint32_t type : 1;
    uint32_t with_imm : 1;
    uint32_t csum_err : 4;
    uint32_t imm_data;
    uint32_t msg_len;
    uint32_t vni;
    uint64_t ts : 48;
    uint64_t wqe_id : 16;
    uint8_t msg_opcode;
    uint8_t rsv0;
    uint16_t rsv1[2];
    uint16_t rsv2 : 15;
    uint16_t owner : 1;
};

struct aclshmemi_xscdv_diamond_cqe_v2 {
    union {
        struct {
            uint32_t error_code : 8;
            uint32_t qp_id : 15;
            uint32_t rsv : 1;
            uint32_t se : 1;
            uint32_t has_pph : 1;
            uint32_t type : 1;
            uint32_t with_imm : 1;
            uint32_t csum_err : 4;
        };
        uint32_t flags_qp_id_err_code;
    };
    uint32_t imm_data;
    uint32_t msg_len;
    uint32_t vni;
    uint32_t ts_l;
    uint32_t ts_h;
    union {
        struct {
            uint32_t msg_opcode : 8;
            uint32_t rsv1 : 4;
            uint32_t wqe_id : 20;
        };
        uint32_t wqe_id_rsv_opcode;
    };
    union {
        struct {
            uint32_t rsv2 : 31;
            uint32_t owner : 1;
        };
        uint32_t owner_rsv;
    };
};

struct aclshmemi_hns_1825_cqe_t {
    uint32_t owner_id_qpn;
    uint32_t op_sr_wqebb;
    uint32_t byte_cnt;
    uint32_t imm_data;
    uint32_t rsvd_dw5;
    uint32_t wqe_num;
    uint32_t vlan_queue_index;
    uint8_t syndrome;
    uint8_t rsvd;
    uint16_t wqe_counter;
};

struct aclshmemi_cqe_ctx {
    uint32_t byte4;
    uint32_t immt_data;
    uint32_t byte12;
    uint32_t byte16;
    uint32_t byte_cnt;
    uint32_t smac;
    uint32_t byte28;
    uint32_t byte32;
};

static_assert(sizeof(aclshmemi_xscdv_diamond_cqe_v1) == 32U, "XSCALE V1 CQE layout changed.");
static_assert(sizeof(aclshmemi_xscdv_diamond_cqe_v2) == 32U, "XSCALE V2 CQE layout changed.");
static_assert(sizeof(aclshmemi_hns_1825_cqe_t) == 32U, "HNS 1825 CQE layout changed.");
static_assert(sizeof(aclshmemi_cqe_ctx) == 32U, "IN_DIE CQE layout changed.");

#endif // ACLSHMEMI_RDMA_CQE_LAYOUT_H
