/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_DEVICE_GM2GM_RMA_H
#define ACLSHMEM_DEVICE_GM2GM_RMA_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/aclshmem_device_mte.h"
#include "device/gm2gm/engine/aclshmem_device_rdma.h"
#include "device/gm2gm/aclshmem_device_mo.h"
#include "host/aclshmem_host_def.h"
#include "device/aclshmem_def.h"
#include "gm2gm/aclshmem_device_rma.hpp"

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
#define ACLSHMEM_TYPE_FUNC(FUNC) \
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

#define ACLSHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                                 \
    /**                                                                                     \
     * @brief Provide a low latency put capability for single element of most basic types.  \
     *                                                                                      \
     * @param dst               [in] Symmetric address of the destination data on local PE. \
     * @param value             [in] The element to be put.                                 \
     * @param pe                [in] The number of the remote PE.                           \
     */                                                                                     \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_p(__gm__ TYPE *dst, const TYPE value, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P_AICORE);

#define ACLSHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                                 \
    /**                                                                                     \
     * @brief Provide a low latency get capability for single element of most basic types.  \
     *                                                                                      \
     * @param src               [in] Symmetric address of the destination data on local PE. \
     * @param pe                [in] The number of the remote PE.                           \
     * @return A single element of type specified in the input pointer.                     \
     */                                                                                     \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_g(__gm__ TYPE *src, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G_AICORE);

/**
 * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                       address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_getmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                       \
    /**                                                                                                          \
     * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to           \
     *                               address on the local PE.                                                    \
     *                                                                                                           \
     * @param dst               [in] Pointer on local device of the destination data.                            \
     * @param src               [in] Pointer on Symmetric memory of the source data.                             \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                           \
     * @param pe                [in] PE number of the remote PE.                                                 \
     */                                                                                                          \
    ACLSHMEM_DEVICE void aclshmem_get_##NAME##_mem(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM);

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                        \
    /**                                                                                                           \
     * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE. \
     *                                                                                                            \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                         \
     * @param src               [in] Pointer on local device of the source data.                                  \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                     \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to
 *                                              address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_getmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

#define ACLSHMEM_GET_TYPENAME_MEM_NBI(NAME, TYPE)                                                                       \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to              \
     *                                                   address on the local PE.                                    \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the             \
     *                 same PE are not supported.                                                                    \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    ACLSHMEM_DEVICE void aclshmem_get_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                            \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on symmetric memory from the specified PE to address on the local device.                            \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_get_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                               \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                              \
    /**                                                                                                            \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to            \
     *                           address on the local PE.                                                          \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the           \
     *                 same PE are not supported.                                                                  \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on local device of the destination data.                         \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                          \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                             \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 uint32_t elem_size, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                     \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on symmetric memory from the specified PE to address on the local device.                            \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on local device of the destination data.                         \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                          \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_NBI(NAME, TYPE)                                                                       \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.     \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the             \
     *                 same PE are not supported.                                                                    \
     *                                                                                                               \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param src               [in] Pointer on local device of the source data.                                     \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                        \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI(NAME, TYPE)                                                            \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on local PE to symmetric address on the specified PE.                                                \
     *                                                                                                             \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                          \
     * @param src               [in] Pointer on local device of the source data.                                   \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                               \
                                                 const non_contiguous_copy_param &copy_params, int32_t pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI(NAME, TYPE)                                                              \
    /**                                                                                                            \
     * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.   \
     *        WARNING: When using RDMA as the underlying transport, concurrent RMA/AMO operations to the           \
     *                 same PE are not supported.                                                                  \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                     \
     * @param src               [in] GlobalTensor on local device of the source data.                              \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                      \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 uint32_t elem_size, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI(NAME, TYPE)                                                     \
    /**                                                                                                            \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                   \
     *        on local PE to symmetric address on the specified PE.                                                \
     *                                                                                                             \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                     \
     * @param src               [in] GlobalTensor on local device of the source data.                              \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.         \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                 const non_contiguous_copy_param &copy_params, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED_NBI);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_nbi(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe);

/**
 * @brief Synchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE
 *       then update sig_addr
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_signal(__gm__ void *dst, __gm__ void *src, size_t elem_size, __gm__ int32_t *sig_addr,
                                      int32_t signal, int sig_op, int pe);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                 \
    /**                                                                                                           \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE. \
     *                                                                                                            \
     * @param dst               [in] Pointer on local device of the destination data.                             \
     * @param src               [in] Pointer on Symmetric memory of the source data.                              \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                            \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                          \
     * @param signal            [in] The value used to update sig_addr.                                           \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:         \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                            \
     * @param pe                [in] PE number of the remote PE.                                                  \
     */                                                                                                           \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,         \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR(NAME, TYPE)                                                              \
    /**                                                                                                               \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.     \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                    size_t elem_size, __gm__ int32_t *sig_addr, int32_t signal,       \
                                                    int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED(NAME, TYPE)                                                         \
    /**                                                                                                            \
     * @brief Synchronous interface. Provide a high-performance way to copy non-contiguous data                    \
     *        on local UB to symmetric address on the specified PE then update sig_addr                            \
     *                                                                                                             \
     * @param dst               [in] Pointer on local device of the destination data.                              \
     * @param src               [in] Pointer on Symmetric memory of the source data.                               \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.       \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                           \
     * @param signal            [in] The value used to update sig_addr.                                            \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:          \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                             \
     * @param pe                [in] PE number of the remote PE.                                                   \
     */                                                                                                            \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal(__gm__ TYPE *dst, __gm__ TYPE *src,                            \
                                                    const non_contiguous_copy_param &copy_params,                  \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED(NAME, TYPE)                                                     \
    /**                                                                                                               \
     * @brief Synchronous interface. Provide a high-performance way to copy non-contiguous data                       \
     *        on local UB to symmetric address on the specified PE.                                                   \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.          \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, \
                                                    const non_contiguous_copy_param &copy_params,                     \
                                                    __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE then update
 * sig_addr
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:
 *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmem_putmem_signal_nbi(__gm__ void *dst, __gm__ void *src, size_t elem_size,
                                          __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                 \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.    \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src, size_t elem_size,         \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI(NAME, TYPE)                                                          \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.    \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                                \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal_nbi(AscendC::GlobalTensor<TYPE> dst,                              \
                                                        AscendC::GlobalTensor<TYPE> src, size_t elem_size,            \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI(NAME, TYPE)                                                        \
    /**                                                                                                               \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                      \
     *        on local UB to symmetric address on the specified PE then update sig_addr                               \
     *                                                                                                                \
     * @param dst               [in] Pointer on local device of the destination data.                                 \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                  \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.          \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                              \
     * @param signal            [in] The value used to update sig_addr.                                               \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:             \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                                \
     * @param pe                [in] PE number of the remote PE.                                                      \
     */                                                                                                               \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal_nbi(__gm__ TYPE *dst, __gm__ TYPE *src,                           \
                                                        const non_contiguous_copy_param &copy_params,                 \
                                                        __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_DETAILED_NBI);

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI(NAME, TYPE)                                               \
    /**                                                                                                             \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                    \
     *        on local UB to symmetric address on the specified PE.                                                 \
     *                                                                                                              \
     * @param dst               [in] Pointer on local device of the destination data.                               \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                \
     * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.        \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                            \
     * @param signal            [in] The value used to update sig_addr.                                             \
     * @param sig_op            [in] Operation used to update sig_addr with signal. Supported operations:           \
     *                               ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                                              \
     * @param pe                [in] PE number of the remote PE.                                                    \
     */                                                                                                             \
    ACLSHMEM_DEVICE void aclshmem_put_##NAME##_mem_signal_nbi(                                                            \
        AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src,                                           \
        const non_contiguous_copy_param &copy_params, __gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe);

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_TENSOR_DETAILED_NBI);

#define ACLSHMEM_TEST(NAME, TYPE)                                                                                \
    /**                                                                                                       \
     * @brief Synchronous interface. Provide a high-performance way to compare data                           \
     *        on local UB to symmetric address on the specified PE.                                           \
     *                                                                                                        \
     * @param ivar               [in] Pointer on local device of the destination data.                        \
     * @param cmp                [in] Pointer on Symmetric memory of the source data.                         \
     * @param cmp_value          [in] Params to describe how non-contiguous data is organized in src and dst. \
     */                                                                                                       \
    ACLSHMEM_DEVICE int aclshmem_##NAME##_test(__gm__ TYPE *ivar, int cmp, TYPE cmp_value);

ACLSHMEM_TEST_TYPE_FUNC(ACLSHMEM_TEST);

#endif