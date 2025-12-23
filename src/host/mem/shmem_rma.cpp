/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "host/data_plane/shmem_host_rma.h"
#include "gm2gm/shmemi_device_rma.h"
#include "host_device/shmem_common_types.h"

using namespace std;
void *aclshmem_ptr(void *ptr, int32_t pe)
{
    if (pe < 0 || pe >= aclshmem_n_pes()) {
        SHM_LOG_ERROR("aclshmem_ptr Failed. PE: " << aclshmem_my_pe() << " Got Ilegal PE !!");
        return nullptr;
    }
    uint64_t lower_bound = (uint64_t)g_state.heap_base;
    uint64_t upper_bound = lower_bound + g_state.heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        SHM_LOG_ERROR("aclshmem_ptr Failed. PE: " << aclshmem_my_pe() << " Got Ilegal Address !!");
        return nullptr;
    }

    uint64_t offset = (uint64_t)ptr - (uint64_t)g_state.heap_base;
    void *symm_ptr = g_state.p2p_device_heap_base[pe];
    if (symm_ptr != nullptr) {
        symm_ptr = reinterpret_cast<void*>(reinterpret_cast<uint64_t>(symm_ptr) + offset);
        return symm_ptr;
    }
    SHM_LOG_ERROR("aclshmem_ptr Failed. PE: " << aclshmem_my_pe()
                                           << " g_state.p2p_device_heap_base contains nullptr, Please Check Init Status!!");
    return nullptr;
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t aclshmemx_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id)
{
    g_state.mte_config.aclshmem_ub = offset;
    g_state.mte_config.ub_size = ub_size;
    g_state.mte_config.event_id = event_id;
    ACLSHMEM_CHECK_RET(update_device_state());
    return ACLSHMEM_SUCCESS;
}

#define ACLSHMEM_TYPE_PUT(NAME, TYPE)                                                                                    \
    /**                                                                                                               \
     * @brief Synchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.     \
     *                                                                                                                \
     * @param dest               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param source             [in] Pointer on local device of the source data.                                     \
     * @param nelems             [in] Number of elements in the destination and source arrays.                        \
     * @param pe                 [in] PE number of the remote PE.                                                     \
     */                                                                                                               \
    ACLSHMEM_HOST_API void aclshmem_put_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe)                       \
    {                                                                                                                 \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dest,      \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,       \
                                              g_state_host.default_stream, g_state_host.default_block_num);           \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed");                                                          \
        }                                                                                                             \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT)
#undef ACLSHMEM_TYPE_PUT

#define ACLSHMEM_TYPE_PUT_NBI(NAME, TYPE)                                                                                \
    /**                                                                                                               \
     * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.    \
     *                                                                                                                \
     * @param dest               [in] Pointer on Symmetric memory of the destination data.                            \
     * @param source             [in] Pointer on local device of the source data.                                     \
     * @param nelems             [in] Number of elements in the destination and source arrays.                        \
     * @param pe                 [in] PE number of the remote PE.                                                     \
     */                                                                                                               \
    ACLSHMEM_HOST_API void aclshmem_put_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)                   \
    {                                                                                                                 \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem_nbi", ACLSHMEMI_OP_PUT, NBI, (uint8_t *)dest,     \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,       \
                                              g_state_host.default_stream, g_state_host.default_block_num);           \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed");                                                          \
        }                                                                                                             \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_PUT_NBI)
#undef ACLSHMEM_TYPE_PUT_NBI

#define ACLSHMEM_TYPE_GET(NAME, TYPE)                                                                                    \
    /**                                                                                                               \
     * @brief Synchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the \
     * local PE.                                                                                                      \
     *                                                                                                                \
     * @param dest               [in] Pointer on local device of the destination data.                                \
     * @param source             [in] Pointer on Symmetric memory of the source data.                                 \
     * @param nelems             [in] Number of elements in the destination and source arrays.                        \
     * @param pe                 [in] PE number of the remote PE.                                                     \
     */                                                                                                               \
    ACLSHMEM_HOST_API void aclshmem_get_##NAME##_mem(TYPE *dest, TYPE *source, size_t nelems, int pe)                       \
    {                                                                                                                 \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_get_" #NAME "_mem", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dest,      \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,       \
                                              g_state_host.default_stream, g_state_host.default_block_num);           \
        if (ret < 0) {                                                                                                \
            SHM_LOG_ERROR("device calling transfer failed");                                                          \
        }                                                                                                             \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET)
#undef ACLSHMEM_TYPE_GET

#define ACLSHMEM_TYPE_GET_NBI(NAME, TYPE)                                                                                 \
    /**                                                                                                                \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the \
     * local PE.                                                                                                       \
     *                                                                                                                 \
     * @param dest               [in] Pointer on local device of the destination data.                                 \
     * @param source             [in] Pointer on Symmetric memory of the source data.                                  \
     * @param nelems             [in] Number of elements in the destination and source arrays.                         \
     * @param pe                 [in] PE number of the remote PE.                                                      \
     */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_get_##NAME##_mem_nbi(TYPE *dest, TYPE *source, size_t nelems, int pe)                    \
    {                                                                                                                  \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_get_" #NAME "_mem_nbi", ACLSHMEMI_OP_GET, NBI, (uint8_t *)dest,      \
                                              (uint8_t *)source, nelems, sizeof(TYPE), pe, nullptr, 0, 0, 1, 1,        \
                                              g_state_host.default_stream, g_state_host.default_block_num);            \
        if (ret < 0) {                                                                                                 \
            SHM_LOG_ERROR("device calling transfer failed");                                                           \
        }                                                                                                              \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPE_GET_NBI)
#undef ACLSHMEM_TYPE_GET_NBI

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL(NAME, TYPE)                                                                    \
    /**                                                                                                              \
     * @brief Synchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.    \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                             \
     * @param signal            [in] The value used to update sig_addr.                                              \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                  \
     *                          Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                              \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    ACLSHMEM_HOST_API void aclshmem_put_##NAME##_mem_signal(TYPE *dst, TYPE *src, size_t elem_size, uint8_t *sig_addr,     \
                                                      int32_t signal, int sig_op, int pe)                            \
    {                                                                                                                \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem_signal", ACLSHMEMI_OP_PUT_SIGNAL, NO_NBI,        \
                                              (uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe, sig_addr, \
                                              signal, sig_op, 1, 1, g_state_host.default_stream,                     \
                                              g_state_host.default_block_num);                                       \
        if (ret < 0) {                                                                                               \
            SHM_LOG_ERROR("device calling transfer failed");                                                         \
        }                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL)
#undef ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL

#define ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI(NAME, TYPE)                                                                \
    /**                                                                                                              \
     * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.   \
     *                                                                                                               \
     * @param dst               [in] Pointer on local device of the destination data.                                \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                 \
     * @param elem_size         [in] Number of elements in the dest and source arrays.                               \
     * @param sig_addr          [in] Symmetric address of the signal word to be updated.                             \
     * @param signal            [in] The value used to update sig_addr.                                              \
     * @param sig_op            [in] Operation used to update sig_addr with signal.                                  \
     *                          Supported operations: ACLSHMEM_SIGNAL_SET/ACLSHMEM_SIGNAL_ADD                              \
     * @param pe                [in] PE number of the remote PE.                                                     \
     */                                                                                                              \
    ACLSHMEM_HOST_API void aclshmem_put_##NAME##_mem_signal_nbi(TYPE *dst, TYPE *src, size_t elem_size, uint8_t *sig_addr, \
                                                          int32_t signal, int sig_op, int pe)                        \
    {                                                                                                                \
        int ret = aclshmemi_prepare_and_post_rma("aclshmem_put_" #NAME "_mem_signal_nbi", ACLSHMEMI_OP_PUT_SIGNAL, NBI,       \
                                              (uint8_t *)dst, (uint8_t *)src, elem_size, sizeof(TYPE), pe, sig_addr, \
                                              signal, sig_op, 1, 1, g_state_host.default_stream,                     \
                                              g_state_host.default_block_num);                                       \
        if (ret < 0) {                                                                                               \
            SHM_LOG_ERROR("device calling transfer failed");                                                         \
        }                                                                                                            \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI)
#undef ACLSHMEM_PUT_TYPENAME_MEM_SIGNAL_NBI

#define ACLSHMEM_TYPENAME_P(NAME, TYPE)                                                                                   \
    /**                                                                                                                \
     * @brief Provide a low latency put capability for single element of most basic types.                             \
     *                                                                                                                 \
     * @param dst               [in] Symmetric address of the destination data on local PE.                            \
     * @param value             [in] The element to be put.                                                            \
     * @param pe                [in] The number of the remote PE.                                                      \
     */                                                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_p(TYPE *dst, const TYPE value, int pe)                                          \
    {                                                                                                                  \
        aclshmemi_prepare_and_post_rma_##NAME##_p("aclshmem_" #NAME "_p", (uint8_t *)dst, value, pe,                         \
                                               g_state_host.default_stream, g_state_host.default_block_num);           \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_P)
#undef ACLSHMEM_TYPENAME_P

#define ACLSHMEM_TYPENAME_G(NAME, TYPE)                                                                                   \
    /**                                                                                                                \
     * @brief Provide a low latency get single element of most basic types.                                            \
     *                                                                                                                 \
     * @param src               [in] Symmetric address of the destination data on local PE.                            \
     * @param pe                [in] The number of the remote PE.                                                      \
     * @return A single element of type specified in the input pointer.                                                \
     */                                                                                                                \
    ACLSHMEM_HOST_API TYPE aclshmem_##NAME##_g(TYPE *src, int32_t pe)                                                        \
    {                                                                                                                  \
        TYPE value {};                                                                                                 \
        auto ptr = aclshmem_ptr(src, pe);                                                                                 \
        if (ptr == nullptr) {                                                                                          \
            SHM_LOG_ERROR("aclshmem_g failed");                                                                           \
            return value;                                                                                              \
        }                                                                                                              \
        int ret =                                                                                                      \
            aclrtMemcpy(&value, sizeof(TYPE), reinterpret_cast<void *>(ptr), sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST); \
        if (ret != 0) {                                                                                                \
            SHM_LOG_ERROR("aclshmem_g failed");                                                                           \
        }                                                                                                              \
        return value;                                                                                                  \
    }

ACLSHMEM_TYPE_FUNC(ACLSHMEM_TYPENAME_G)
#undef ACLSHMEM_TYPENAME_G

void aclshmem_putmem(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("shmem putmem", ACLSHMEMI_OP_PUT, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_putmem failed");
    }
}

void aclshmem_getmem(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("shmem getmem", ACLSHMEMI_OP_GET, NO_NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_getmem failed");
    }
}

void aclshmem_putmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmem_putmem_nbi", ACLSHMEMI_OP_PUT, NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_putmem_nbi failed");
    }
}

void aclshmem_getmem_nbi(void *dst, void *src, size_t elem_size, int32_t pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmem_getmem_nbi", ACLSHMEMI_OP_GET, NBI, (uint8_t *)dst, (uint8_t *)src,
                                          elem_size, 1, pe, nullptr, 0, 0, 1, 1, g_state_host.default_stream,
                                          g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmem_getmem_nbi failed");
    }
}

void aclshmemx_putmem_signal_nbi(void *dst, void *src, size_t elem_size, void *sig_addr, int32_t signal, int sig_op, int pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_putmem_signal_nbi", ACLSHMEMI_OP_PUT_SIGNAL, NBI, (uint8_t *)dst,
                                          (uint8_t *)src, elem_size, 1, pe, (uint8_t *)sig_addr, signal, sig_op, 1, 1,
                                          g_state_host.default_stream, g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("device calling transfer failed");
    }
}

void aclshmemx_putmem_signal(void *dst, void *src, size_t elem_size, void *sig_addr, int32_t signal, int sig_op, int pe)
{
    int ret = aclshmemi_prepare_and_post_rma("aclshmemx_putmem_signal", ACLSHMEMI_OP_PUT_SIGNAL, NO_NBI, (uint8_t *)dst,
                                          (uint8_t *)src, elem_size, 1, pe, (uint8_t *)sig_addr, signal, sig_op, 1, 1,
                                          g_state_host.default_stream, g_state_host.default_block_num);
    if (ret < 0) {
        SHM_LOG_ERROR("device calling transfer failed");
    }
}

void aclshmemx_getmem_on_stream(void* dst, void* src, size_t elem_size, int32_t pe, aclrtStream stream)
{
    int ret = aclshmemi_getmem_on_stream((uint8_t *)dst, (uint8_t *)src, elem_size, pe, stream);
    if (ret < 0) {
        SHM_LOG_ERROR("aclshmemi_getmem_on_stream failed");
    }
}
