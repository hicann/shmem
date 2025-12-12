/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_UB2GM_RMA_H
#define SHMEM_DEVICE_UB2GM_RMA_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/aclshmem_device_mte.h"
#include "internal/device/shmemi_device_common.h"
#include "internal/device/sync/shmemi_device_p2p.h"
#include "device/aclshmem_def.h"

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
 */
#define SHMEM_TYPE_FUNC(FUNC) \
    FUNC(half, half);         \
    FUNC(float, float);       \
    FUNC(double, double);     \
    FUNC(int8, int8_t);       \
    FUNC(int16, int16_t);     \
    FUNC(int32, int32_t);     \
    FUNC(int64, int64_t);     \
    FUNC(uint8, uint8_t);     \
    FUNC(uint16, uint16_t);   \
    FUNC(uint32, uint32_t);   \
    FUNC(uint64, uint64_t);   \
    FUNC(char, char);         \
    FUNC(bfloat16, bfloat16_t)

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE
 *        to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__ubuf__ T *dst, __gm__ T *src, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    shmemi_copy_gm2ub(dst, remote_ptr, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *        address on the local UB.
 *
 * @param dst               [in] LocalTensor on local UB of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t elem_size,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    shmemi_copy_gm2ub(dst, remote_buff, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local UB.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__ubuf__ T *dst, __gm__ T *src, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst);
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.dst_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dst);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (copy_params.dst_ld - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    shmemi_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local UB.
 *
 * @param dst               [in] LocalTensor on local UB of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.dst_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (copy_params.dst_ld - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    shmemi_copy_gm2ub(dst, remote_buff, data_copy_params_gm2ub);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __ubuf__ T *src, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit)
        return;

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    shmemi_copy_ub2gm(remote_ptr, src, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, uint32_t elem_size,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = elem_size;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    shmemi_copy_ub2gm(remote_buff, src, elem_size * sizeof(T));
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __ubuf__ T *src, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src);
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.src_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(src);
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    shmemi_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] LocalTensor on local UB of the source data.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    // Check if Process will out of UB address.
    uint64_t ub_offset = reinterpret_cast<uint64_t>(src.GetPhyAddr());
    uint64_t process_num = (copy_params.repeat - 1) * copy_params.src_ld + copy_params.length;
    if (ub_offset + process_num * sizeof(T) > ub_limit) {
        return;
    }

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    shmemi_copy_ub2gm(remote_buff, src, data_copy_params_ub2gm);
}

#define SHMEM_GET_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                  \
    /**                                                                                                            \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the                            \
     *          specified PE to address on the local UB.                                                           \
     *                                                                                                             \
     * @param dst               [in] Pointer on local UB of the destination data.                                  \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                      \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int pe) \
    {                                                                                                              \
        /* MTE  */                                                                                                 \
        /* Global State Get */                                                                                     \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                      \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                    \
        shmem_mte_get_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                          \
    /**                                                                                                           \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to           \
     *          address on the local UB.                                                                          \
     *                                                                                                            \
     * @param dst               [in] LocalTensor on local UB of the destination data.                             \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                         \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 uint32_t elem_size, int pe)                                      \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_get_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_TENSOR_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                                 \
    /**                                                                                                    \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data           \
     *        on symmetric memory from the specified PE to address on the local UB.                        \
     *                                                                                                     \
     * @param dst               [in] Pointer on local UB of the destination data.                          \
     * @param src               [in] Pointer on Symmetric memory of the source data.                       \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst. \
     * @param pe                [in] PE number of the remote PE.                                           \
     */                                                                                                    \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__ubuf__ TYPE *dst, __gm__ TYPE *src,                     \
                                                 const non_contiguous_copy_param &copy_params, int pe)     \
    {                                                                                                      \
        /* MTE  */                                                                                         \
        /* Global State Get */                                                                             \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                              \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;            \
        shmem_mte_get_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                   \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_DETAILED_NBI);

#define SHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                 \
    /**                                                                                                           \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                  \
     *        on symmetric memory from the specified PE to address on the local UB.                               \
     *                                                                                                            \
     * @param dst               [in] LocalTensor on local UB of the destination data.                             \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                         \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.        \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int pe)            \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_get_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_NBI(NAME, TYPE)                                                                      \
    /**                                                                                                                \
     * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.       \
     *                                                                                                                 \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                              \
     * @param src               [in] Pointer on local UB of the source data.                                           \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                          \
     * @param pe                [in] PE number of the remote PE.                                                       \
     */                                                                                                                \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src, uint32_t elem_size, int32_t pe) \
    {                                                                                                                  \
        /* MTE  */                                                                                                     \
        /* Global State Get */                                                                                         \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                        \
        shmem_mte_put_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                                 \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI(NAME, TYPE)                                                          \
    /**                                                                                                           \
     * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.  \
     *                                                                                                            \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                    \
     * @param src               [in] LocalTensor on local UB of the source data.                                  \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, \
                                                 uint32_t elem_size, int32_t pe)                                  \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_put_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI(NAME, TYPE)                                                   \
    /**                                                                                                      \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data             \
     *        on local UB to symmetric address on the specified PE.                                          \
     *                                                                                                       \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                    \
     * @param src               [in] Pointer on local UB of the source data.                                 \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst. \
     * @param pe                [in] PE number of the remote PE.                                             \
     */                                                                                                      \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __ubuf__ TYPE *src,                       \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)   \
    {                                                                                                        \
        /* MTE  */                                                                                           \
        /* Global State Get */                                                                               \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;              \
        shmem_mte_put_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                     \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_DETAILED_NBI);

#define SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI(NAME, TYPE)                                                 \
    /**                                                                                                           \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                  \
     *        on local UB to symmetric address on the specified PE.                                               \
     *                                                                                                            \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                    \
     * @param src               [in] LocalTensor on local UB of the source data.                                  \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.      \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe)        \
    {                                                                                                             \
        /* MTE  */                                                                                                \
        /* Global State Get */                                                                                    \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                     \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                   \
        shmem_mte_put_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED_NBI);

#endif
