/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_SDMA_H
#define SHMEM_DEVICE_SDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmem_device_sdma.hpp"

/**
 * @brief Configure the UB workspace and pipeline synchronization ID used by high-level SDMA RMA interfaces.
 *
 * @param offset             [in] Start address of the UB workspace. The address must be 64-byte aligned.
 * @param ub_size            [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param sync_id            [in] Hardware event ID used for pipeline synchronization.
 * @note This function configures the implicit workspace used by high-level aclshmem_* RMA APIs. The low-level SDMA
 *       interfaces below receive their workspace explicitly.
 */
ACLSHMEM_DEVICE void aclshmemx_set_sdma_config(uint64_t offset, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory using
 *        SDMA QP 0.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] Local GM destination address.
 * @param src              [in] Symmetric source address; it is translated to the corresponding address on @p pe.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Source PE. It must be in the initialized PE range.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note The complete source range must remain within one symmetric allocation and the destination range must be
 *       valid local GM. A normal return means the SDMA request was submitted, not completed. Call
 *       aclshmemx_sdma_quiet from the submitting block before reading @p dst or reusing the workspace.
 * @note This single-core interface always uses QP 0. Use aclshmemx_sdma_qp_get_nbi for multi-core submissions.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief GlobalTensor/LocalTensor overload of aclshmemx_sdma_get_nbi.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] GlobalTensor backed by local GM for the destination data.
 * @param src              [in] GlobalTensor at a symmetric source address; it is translated to @p pe.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Source PE. It must be in the initialized PE range.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note Address, QP 0, and completion requirements are the same as the pointer overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(
    AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src, AscendC::LocalTensor<T>& buf, uint32_t elem_size,
    int pe, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE using
 *        an explicitly selected SDMA QP.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] Symmetric destination address; it is translated to the corresponding address on
 *                          @p pe.
 * @param src              [in] Local GM source address.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Destination PE. It must be in the initialized PE range.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note The complete destination range must remain within one symmetric allocation and the source range must be
 *       valid local GM. A normal return means the request was submitted, not completed. Keep @p src and the
 *       workspace unchanged until aclshmemx_sdma_qp_quiet with the same @p qp_idx returns.
 * @note Do not wait on aclshmemx_sdma_quiet here: it only completes QP 0 and does not complete requests submitted
 *       with @p qp_idx > 0. Complete this request with aclshmemx_sdma_qp_quiet using the same @p qp_idx, or append
 *       a notify record with aclshmemx_sdma_qp_notify_record using the same @p qp_idx. The QP index is independent
 *       of the block index.
 * @note SDMA write is not supported on Ascend950.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t qp_idx,
    uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from symmetric memory on the specified PE to local device memory using
 *        an explicitly selected SDMA QP.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] Local GM destination address.
 * @param src              [in] Symmetric source address; it is translated to the corresponding address on @p pe.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Source PE. It must be in the initialized PE range.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note The complete source range must remain within one symmetric allocation and the destination range must be
 *       valid local GM. A normal return means the request was submitted, not completed. Call aclshmemx_sdma_qp_quiet
 *       with the same @p qp_idx before reading @p dst or reusing the workspace.
 * @note Do not wait on aclshmemx_sdma_quiet here: it only completes QP 0 and does not complete requests submitted
 *       with @p qp_idx > 0. Complete this request with aclshmemx_sdma_qp_quiet using the same @p qp_idx, or append
 *       a notify record with aclshmemx_sdma_qp_notify_record using the same @p qp_idx. The QP index is independent
 *       of the block index.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t qp_idx,
    uint32_t sync_id);

/**
 * @brief GlobalTensor/LocalTensor overload of aclshmemx_sdma_qp_put_nbi.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] GlobalTensor at a symmetric destination address; it is translated to @p pe.
 * @param src              [in] GlobalTensor backed by local GM for the source data.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Destination PE. It must be in the initialized PE range.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note Address, completion, explicit QP, and Ascend950 restrictions are the same as the pointer overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_put_nbi(
    AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src, AscendC::LocalTensor<T>& buf, uint32_t elem_size,
    int pe, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief GlobalTensor/LocalTensor overload of aclshmemx_sdma_qp_get_nbi.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] GlobalTensor backed by local GM for the destination data.
 * @param src              [in] GlobalTensor at a symmetric source address; it is translated to @p pe.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Source PE. It must be in the initialized PE range.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note Address, completion, and explicit QP requirements are the same as the pointer overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_get_nbi(
    AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src, AscendC::LocalTensor<T>& buf, uint32_t elem_size,
    int pe, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Asynchronously copy contiguous data from local device memory to symmetric memory on the specified PE using
 *        SDMA QP 0.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] Symmetric destination address; it is translated to the corresponding address on
 *                          @p pe.
 * @param src              [in] Local GM source address.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Destination PE. It must be in the initialized PE range.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note The complete destination range must remain within one symmetric allocation and the source range must be
 *       valid local GM. A normal return means the request was submitted, not completed. Keep @p src unchanged until
 *       aclshmemx_sdma_quiet returns.
 * @note This single-core interface always uses QP 0. Use aclshmemx_sdma_qp_put_nbi for multi-core submissions.
 * @note SDMA write is not supported on Ascend950.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief GlobalTensor/LocalTensor overload of aclshmemx_sdma_put_nbi.
 *
 * @tparam T               Element type of the transfer.
 * @param dst              [in] GlobalTensor at a symmetric destination address; it is translated to @p pe.
 * @param src              [in] GlobalTensor backed by local GM for the source data.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param elem_size        [in] Number of T elements to transfer; elem_size * sizeof(T) must not exceed
 *                          UINT32_MAX bytes.
 * @param pe               [in] Destination PE. It must be in the initialized PE range.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note Address, QP 0, completion, and Ascend950 restrictions are the same as the pointer overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(
    AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src, AscendC::LocalTensor<T>& buf, uint32_t elem_size,
    int pe, uint32_t sync_id);

/**
 * @brief Asynchronously submit an L2 cache maintenance operation for device global memory.
 *
 * @tparam T               Element type used to calculate the operated byte range.
 * @param src              [in] Local GM start address of the range to operate on.
 * @param elem_size        [in] Number of T elements in the operated range.
 * @param cmo_type         [in] Cache operation type. Only ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH is currently accepted;
 *                          other values return without submitting an operation.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note This interface submits to QP 0 and is non-blocking. Call aclshmemx_sdma_quiet before relying on completion.
 *       Use aclshmemx_cmo_qp_nbi for an explicitly selected QP.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(
    __gm__ T* src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type, __ubuf__ T* buf, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief GlobalTensor/LocalTensor overload of aclshmemx_cmo_nbi.
 *
 * @tparam T               Element type used to calculate the operated byte range.
 * @param src              [in] GlobalTensor backed by local GM for the operated range.
 * @param elem_size        [in] Number of T elements in the operated range.
 * @param cmo_type         [in] Cache operation type. Only ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH is currently accepted.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note Completion requirements are the same as the pointer overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_nbi(
    AscendC::GlobalTensor<T>& src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type, AscendC::LocalTensor<T>& buf,
    uint32_t sync_id);

/**
 * @brief Asynchronously submit an L2 cache maintenance operation to an explicitly selected SDMA QP.
 *
 * @tparam T               Element type used to calculate the operated byte range.
 * @param src              [in] Local GM start address of the range to operate on.
 * @param elem_size        [in] Number of T elements in the operated range.
 * @param cmo_type         [in] Cache operation type. Only ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH is currently accepted;
 *                          other values return without submitting an operation.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note This interface is non-blocking. Complete it with aclshmemx_sdma_qp_quiet using the same @p qp_idx.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_qp_nbi(
    __gm__ T* src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type, __ubuf__ T* buf, uint32_t ub_size, uint32_t qp_idx,
    uint32_t sync_id);

/**
 * @brief GlobalTensor/LocalTensor overload of aclshmemx_cmo_qp_nbi.
 *
 * @tparam T               Element type used to calculate the operated byte range.
 * @param src              [in] GlobalTensor backed by local GM for the operated range.
 * @param elem_size        [in] Number of T elements in the operated range.
 * @param cmo_type         [in] Cache operation type. Only ACLSHMEMCMOTYPE::CMO_TYPE_PREFETCH is currently accepted.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note QP scope and completion requirements are the same as the pointer overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_cmo_qp_nbi(
    AscendC::GlobalTensor<T>& src, uint32_t elem_size, ACLSHMEMCMOTYPE cmo_type, AscendC::LocalTensor<T>& buf,
    uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Wait until all SDMA SQEs previously submitted to QP 0 have completed.
 *
 * @tparam T               Element type of the LocalTensor workspace.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note This function only drains QP 0 and is the completion operation paired with aclshmemx_sdma_put_nbi and
 *       aclshmemx_sdma_get_nbi. Use aclshmemx_sdma_qp_quiet for an explicitly selected QP.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(AscendC::LocalTensor<T>& buf, uint32_t sync_id);

/**
 * @brief Pointer overload of aclshmemx_sdma_quiet.
 *
 * @tparam T               Element type of the UB workspace.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note QP scope and completion requirements are the same as the LocalTensor overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_quiet(__ubuf__ T* buf, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Wait until all SDMA SQEs previously submitted to an explicitly selected QP have completed.
 *
 * @tparam T               Element type of the LocalTensor workspace.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note Use the same @p qp_idx as the preceding aclshmemx_sdma_qp_put_nbi or aclshmemx_sdma_qp_get_nbi calls.
 *       This function waits for one QP only and does not derive the QP from AscendC::GetBlockIdx().
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_quiet(AscendC::LocalTensor<T>& buf, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Pointer overload of aclshmemx_sdma_qp_quiet.
 *
 * @tparam T               Element type of the UB workspace.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note QP scope and completion requirements are the same as the LocalTensor overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_quiet(__ubuf__ T* buf, uint32_t ub_size, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Submit a STARS notify-record SQE on SDMA QP 0.
 *
 * @tparam T               Element type of the LocalTensor workspace.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note The notify record is ordered after earlier SQEs on QP 0. Host code can wait for the configured notify ID with
 *       aclrtWaitAndResetNotify. Use aclshmemx_sdma_qp_notify_record for an explicitly selected QP.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_notify_record(AscendC::LocalTensor<T>& buf, uint32_t sync_id);

/**
 * @brief Pointer overload of aclshmemx_sdma_notify_record.
 *
 * @tparam T               Element type of the UB workspace.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note QP scope and ordering requirements are the same as the LocalTensor overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_notify_record(__ubuf__ T* buf, uint32_t ub_size, uint32_t sync_id);

/**
 * @brief Submit a STARS notify-record SQE on an explicitly selected SDMA QP.
 *
 * @tparam T               Element type of the LocalTensor workspace.
 * @param buf              [in] LocalTensor backed by at least 64 bytes of local UB workspace.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note The notify record is ordered after earlier SQEs on the same @p qp_idx. Use the same QP as the corresponding
 *       aclshmemx_sdma_qp_put_nbi or aclshmemx_sdma_qp_get_nbi calls.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_notify_record(AscendC::LocalTensor<T>& buf, uint32_t qp_idx, uint32_t sync_id);

/**
 * @brief Pointer overload of aclshmemx_sdma_qp_notify_record.
 *
 * @tparam T               Element type of the UB workspace.
 * @param buf              [in] Local UB workspace. The address must be 64-byte aligned.
 * @param ub_size          [in] UB workspace size in bytes. It must be at least 64 bytes.
 * @param qp_idx           [in] SDMA QP index. It must be smaller than the configured SDMA channel count.
 * @param sync_id          [in] Hardware event ID used for pipeline synchronization.
 * @note QP scope and ordering requirements are the same as the LocalTensor overload.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_qp_notify_record(
    __ubuf__ T* buf, uint32_t ub_size, uint32_t qp_idx, uint32_t sync_id);

#endif
