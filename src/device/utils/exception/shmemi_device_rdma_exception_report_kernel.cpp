/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/exception/shmemi_device_rdma_exception_report_kernel.h"

#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"
#include "host_device/shmemi_rdma_cqe_layout.h"
#include "device/gm2gm/engine/shmemi_device_rdma.h"

namespace {
constexpr uint64_t kDataCacheLineSize = 64U;

ACLSHMEM_DEVICE bool InvalidateDataCache(__gm__ uint8_t* addr, uint64_t size)
{
    if (addr == nullptr || size == 0U) {
        return false;
    }
    const uint64_t address = reinterpret_cast<uint64_t>(addr);
    if (address > UINT64_MAX - (size - 1U)) {
        return false;
    }
    const uint64_t begin = address / kDataCacheLineSize * kDataCacheLineSize;
    const uint64_t end = (address + size - 1U) / kDataCacheLineSize * kDataCacheLineSize;
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(begin));
    for (uint64_t offset = 0; offset <= end - begin; offset += kDataCacheLineSize) {
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[offset]);
        __asm__ __volatile__("");
    }
    return true;
}

ACLSHMEM_DEVICE bool ReadRaw(__gm__ uint8_t* source, uint64_t size, __gm__ aclshmemi_rdma_exception_report_entry_t* out)
{
    if (source == nullptr || size == 0U || size > ACLSHMEMI_RDMA_EXCEPTION_REPORT_MAX_RAW_SIZE) {
        return false;
    }
    if (!InvalidateDataCache(source, size)) {
        return false;
    }
    out->raw.addr = reinterpret_cast<uint64_t>(source);
    out->raw.size = static_cast<uint32_t>(size);
    for (uint32_t i = 0; i < size; ++i) {
        out->raw.data[i] = source[i];
    }
    return true;
}

ACLSHMEM_DEVICE bool ReadCqe(__gm__ uint8_t* source, uint64_t size, __gm__ aclshmemi_rdma_exception_report_entry_t* out)
{
    if (size < sizeof(aclshmemi_cqe_ctx) || !ReadRaw(source, size, out)) {
        return false;
    }
#if defined(ACLSHMEMI_RDMA_K_BACKEND_XSCALE)
    out->cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID;
    if constexpr (ACLSHMEMI_XSCALE_API_VERSION_VAR == 1) {
        auto* cqe = reinterpret_cast<__gm__ aclshmemi_xscdv_diamond_cqe_v1*>(source);
        out->cqe.owner = cqe->owner;
        out->cqe.status = cqe->error_code;
        out->cqe.wqn = cqe->qp_id;
        out->cqe.opcode = cqe->msg_opcode;
        out->cqe.wqe_id = cqe->wqe_id;
    } else {
        auto* cqe = reinterpret_cast<__gm__ aclshmemi_xscdv_diamond_cqe_v2*>(source);
        out->cqe.owner = cqe->owner;
        out->cqe.status = cqe->error_code;
        out->cqe.wqn = cqe->qp_id;
        out->cqe.opcode = cqe->msg_opcode;
        out->cqe.wqe_id = cqe->wqe_id;
    }
#elif defined(ACLSHMEMI_RDMA_K_BACKEND_HNS_1825)
    auto* cqe = reinterpret_cast<__gm__ aclshmemi_hns_1825_cqe_t*>(source);
    out->cqe.valid_fields =
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OPCODE_VALID |
        ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_SYNDROME_VALID | ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQE_ID_VALID;
    out->cqe.owner = (cqe->owner_id_qpn >> 31U) & 1U;
    out->cqe.wqn = cqe->owner_id_qpn & 0xfffffU;
    out->cqe.opcode = (cqe->op_sr_wqebb >> 27U) & 0x1fU;
    out->cqe.syndrome = cqe->syndrome;
    out->cqe.status = out->cqe.opcode == 0x1eU ? out->cqe.syndrome : 0U;
    out->cqe.wqe_id = cqe->wqe_counter;
#else
    auto* cqe = reinterpret_cast<__gm__ aclshmemi_cqe_ctx*>(source);
    out->cqe.valid_fields = ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_OWNER_VALID |
                            ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_STATUS_VALID |
                            ACLSHMEMI_RDMA_EXCEPTION_REPORT_CQE_WQN_VALID;
    out->cqe.owner = (cqe->byte4 >> 7U) & 1U;
    out->cqe.status = (cqe->byte4 >> 8U) & 0xffU;
    out->cqe.wqn = cqe->byte16 & 0xffffffU;
#endif
    return true;
}

ACLSHMEM_DEVICE bool ReadWq(
    __gm__ aclshmemi_rdma_sq_ctx* source, uint64_t size, __gm__ aclshmemi_rdma_exception_report_entry_t* out)
{
    if (source == nullptr || size < sizeof(aclshmemi_rdma_sq_ctx)) {
        return false;
    }
    if (!InvalidateDataCache(reinterpret_cast<__gm__ uint8_t*>(source), sizeof(aclshmemi_rdma_sq_ctx))) {
        return false;
    }
    out->wq.wqn = source->wqn;
    out->wq.buf_addr = source->buf_addr;
    out->wq.wqe_size = source->wqe_size;
    out->wq.depth = source->depth;
    out->wq.head_addr = source->head_addr;
    out->wq.tail_addr = source->tail_addr;
    out->wq.db_mode = static_cast<int32_t>(source->db_mode);
    out->wq.db_addr = source->db_addr;
    out->wq.sl = source->sl;
    out->wq.amo_addr = source->amo_addr;
    out->wq.amo_lkey = source->amo_lkey;
    out->wq.db_sw_addr = source->db_sw_addr;
    out->wq.mtu_shift = source->mtu_shift;
    out->wq.db_cos = source->db_cos;
    return true;
}

ACLSHMEM_DEVICE bool ReadCq(
    __gm__ aclshmemi_rdma_cq_ctx* source, uint64_t size, __gm__ aclshmemi_rdma_exception_report_entry_t* out)
{
    if (source == nullptr || size < sizeof(aclshmemi_rdma_cq_ctx)) {
        return false;
    }
    if (!InvalidateDataCache(reinterpret_cast<__gm__ uint8_t*>(source), sizeof(aclshmemi_rdma_cq_ctx))) {
        return false;
    }
    out->cq.cqn = source->cqn;
    out->cq.buf_addr = source->buf_addr;
    out->cq.cqe_size = source->cqe_size;
    out->cq.depth = source->depth;
    out->cq.head_addr = source->head_addr;
    out->cq.tail_addr = source->tail_addr;
    out->cq.db_mode = static_cast<int32_t>(source->db_mode);
    out->cq.db_addr = source->db_addr;
    out->cq.cq_attr_flags = source->cq_attr_flags;
    out->cq.db_sw_addr = source->db_sw_addr;
    return true;
}

} // namespace

ACLSHMEM_GLOBAL void aclshmemi_rdma_exception_report_read_kernel(
    uint32_t entry_type, GM_ADDR addr, uint64_t size, GM_ADDR out_addr)
{
    auto* out = reinterpret_cast<__gm__ aclshmemi_rdma_exception_report_entry_t*>(out_addr);
    if (out == nullptr) {
        return;
    }
    out->ret = ACLSHMEMI_RDMA_EXCEPTION_REPORT_INVALID_PARAM;
    out->entry_type = entry_type;
    if (addr == nullptr || size == 0U) {
        return;
    }
    bool success = false;
    if (entry_type == ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_WQ) {
        success = ReadWq(reinterpret_cast<__gm__ aclshmemi_rdma_sq_ctx*>(addr), size, out);
    } else if (entry_type == ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQ) {
        success = ReadCq(reinterpret_cast<__gm__ aclshmemi_rdma_cq_ctx*>(addr), size, out);
    } else if (entry_type == ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_CQE_RAW) {
        success = ReadCqe(reinterpret_cast<__gm__ uint8_t*>(addr), size, out);
    } else if (entry_type == ACLSHMEMI_RDMA_EXCEPTION_REPORT_ENTRY_RAW) {
        success = ReadRaw(reinterpret_cast<__gm__ uint8_t*>(addr), size, out);
    }
    if (success) {
        out->ret = ACLSHMEMI_RDMA_EXCEPTION_REPORT_SUCCESS;
    }
}

int32_t aclshmemi_rdma_exception_report_read_entry_on_stream(
    uint32_t entry_type, uint64_t addr, uint64_t size, aclshmemi_rdma_exception_report_entry_t* out, aclrtStream stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    aclshmemi_rdma_exception_report_read_kernel<<<1, nullptr, stream>>>(entry_type, (GM_ADDR)addr, size, (GM_ADDR)out);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
