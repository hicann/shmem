/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_MTE_H
#define SHMEM_DEVICE_MTE_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "device/aclshmem_def.h"

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] Pointer on Symmetric memory of the destination data.
 * @param srcUb             [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Byte Size of data in the destination and source arrays.
 * @param toL2Cache         [in] Enable L2Cache or not. False means disable L2Cache.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    uint32_t size, bool toL2Cache = true)
{
    ASCENDC_ASSERT((dstGva != nullptr), "input gva is null");

    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(srcUb);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGva));
    if (!toL2Cache) {
        gmTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] GlobalTensor on Symmetric memory of the destination data.
 * @param srcUb             [in] LocalTensor on local UB of the source data.
 * @param size              [in] Byte Size of data in the destination and source arrays.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_ub2gm(const AscendC::GlobalTensor<T> &dstGva,
    const AscendC::LocalTensor<T> &srcUb, uint32_t size)
{
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    AscendC::DataCopyPad(dstGva, srcUb, dataCopyParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] Pointer on Symmetric memory of the destination data.
 * @param srcUb             [in] Pointer on local UB of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_ub2gm(__gm__ T* dstGva, __ubuf__ T* srcUb,
    AscendC::DataCopyExtParams &copyParams)
{
    ASCENDC_ASSERT((dstGva != nullptr), "input gva is null");

    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(srcUb);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGva));
    AscendC::DataCopyPad(gmTensor, ubTensor, copyParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstGva            [in] GlobalTensor on Symmetric memory of the destination data.
 * @param srcUb             [in] LocalTensor on local UB of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_ub2gm(const AscendC::GlobalTensor<T> &dstGva,
    const AscendC::LocalTensor<T> &srcUb, AscendC::DataCopyExtParams &copyParams)
{
    AscendC::DataCopyPad(dstGva, srcUb, copyParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on symmetric memory from the specified PE to local UB.
 *
 * @param dstUb             [in] Pointer on local UB of the destination data.
 * @param srcGva            [in] Pointer on Symmetric memory of the source data.
 * @param size              [in] Byte Size of data in the destination and source arrays.
 * @param toL2Cache         [in] Enable L2Cache or not. False means disable L2Cache.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    uint32_t size, bool toL2Cache = true)
{
    ASCENDC_ASSERT((srcGva != nullptr), "input gva is null");
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dstUb);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGva));
    if (!toL2Cache) {
        gmTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on symmetric memory from the specified PE to local UB.
 *
 * @param dstUb             [in] LocalTensor on local UB of the destination data.
 * @param srcGva            [in] GlobalTensor on Symmetric memory of the source data.
 * @param size              [in] Byte Size of data in the destination and source arrays.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_gm2ub(const AscendC::LocalTensor<T> &dstUb,
    const AscendC::GlobalTensor<T> &srcGva, uint32_t size)
{
    AscendC::DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(dstUb, srcGva, dataCopyParams, padParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstUb             [in] Pointer on local UB of the destination data.
 * @param srcGva            [in] Pointer on Symmetric memory of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_gm2ub(__ubuf__ T* dstUb, __gm__ T* srcGva,
    AscendC::DataCopyExtParams &copyParams)
{
    ASCENDC_ASSERT((srcGva != nullptr), "input gva is null");
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> gmTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(dstUb);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGva));
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(ubTensor, gmTensor, copyParams, padParams);
}

/**
 * @brief Simple Copy interface. Copy contiguous data on local UB memory to symmetric address on the specified PE.
 *
 * @param dstUb             [in] LocalTensor on local UB of the destination data.
 * @param srcGva            [in] GlobalTensor on Symmetric memory of the source data.
 * @param copyParams        [in] Describe non-contiguous data copy.
 */
template<typename T>
SHMEM_DEVICE void shmemi_copy_gm2ub(const AscendC::LocalTensor<T> &dstUb,
    const AscendC::GlobalTensor<T> &srcGva, AscendC::DataCopyExtParams &copyParams)
{
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(dstUb, srcGva, copyParams, padParams);
}

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return A remote symmetric address on the specified PE that can be accessed using memory loads and stores.
 */
SHMEM_DEVICE __gm__ void *shmem_ptr(__gm__ void *ptr, int pe)
{
    // Get Global State
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);

    // Address translate
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->device_p2p_heap_base[pe]) + offset;

    return reinterpret_cast<__gm__ void *>(remote_ptr);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        shmemi_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(dst + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        shmemi_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    shmemi_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    shmemi_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 *        PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    // block_size: dataMove Unit
    uint64_t block_size = buf.GetSize() * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        shmemi_copy_gm2ub(buf, remote_buff[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(dst[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        shmemi_copy_gm2ub(buf, remote_buff[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(dst[repeat_times * repeat_elem], buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    shmemi_copy_gm2ub(buf, remote_buff, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    shmemi_copy_ub2gm(dst, buf, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        shmemi_copy_gm2ub(buf, src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        shmemi_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                        const non_contiguous_copy_param &copy_params, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(remote_ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    shmemi_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    shmemi_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe,
                                        AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    // block_size: dataMove Unit
    uint64_t block_size = buf.GetSize() * sizeof(T);
    uint64_t remain = (elem_size * sizeof(T)) % block_size;

    uint64_t repeat_times = (elem_size * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        shmemi_copy_gm2ub(buf, src[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(remote_buff[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {  // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        shmemi_copy_gm2ub(buf, src[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(remote_buff[repeat_times * repeat_elem], buf, remain);
    }
}

/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src,
                                        AscendC::LocalTensor<T> buf, const non_contiguous_copy_param &copy_params,
                                        int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr));

    uint64_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint64_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (copy_params.src_ld - copy_params.length) * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT, 0);
    shmemi_copy_gm2ub(buf, src, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(copy_params.repeat, copy_params.length * sizeof(T),
                                                      (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
                                                      (copy_params.dst_ld - copy_params.length) * sizeof(T), 0);
    shmemi_copy_ub2gm(remote_buff, buf, data_copy_params_ub2gm);
}

/**
 * @brief Async interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 * @param enable_L2         [in] whether to enable L2 cache.
 */
SHMEM_DEVICE void shmemx_mte_get_mem_nbi_low_level(__gm__ int8_t* dst, __gm__ int8_t* src, __ubuf__ int8_t* buf,
    uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID, bool enable_L2)
{
    auto ptr = shmem_ptr(src, pe);
    __gm__ int8_t* remote_ptr = reinterpret_cast<__gm__ int8_t*>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size;
    uint64_t remain = (elem_size) % block_size;

    uint64_t repeat_times = (elem_size) / block_size;
    uint64_t repeat_elem = block_size;
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        shmemi_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(dst + i * repeat_elem, buf, block_size, enable_L2);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        shmemi_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain, enable_L2);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size           [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 * @param enable_L2         [in] whether to enable L2 cache.
 */
SHMEM_DEVICE void shmemx_mte_put_mem_nbi_low_level(__gm__ int8_t* dst, __gm__ int8_t* src, __ubuf__ int8_t* buf,
    uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID, bool enable_L2)
{
    auto ptr = shmem_ptr(dst, pe);
    __gm__ int8_t* remote_ptr = reinterpret_cast<__gm__ int8_t*>(ptr);

    // block_size: dataMove Unit
    uint64_t block_size = ub_size;
    uint64_t remain = (elem_size) % block_size;

    uint64_t repeat_times = (elem_size) / block_size;
    uint64_t repeat_elem = block_size;
    uint64_t loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (uint64_t i = 0; i < repeat_times; i++) {
        shmemi_copy_gm2ub(buf, src + i * repeat_elem, block_size, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size, enable_L2);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        shmemi_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain, enable_L2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        shmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain, enable_L2);
    }
}

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address
 *        on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param enable_L2         [in] whether to enable L2 cache
 */
SHMEM_DEVICE void shmemx_mte_get_mem_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, uint32_t elem_size,
    int32_t pe, bool enable_L2)
{
    /* Global State Get */
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.shmem_ub;
    uint32_t copy_ub_size = device_state->mte_config.ub_size;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
    shmemx_mte_get_mem_nbi_low_level(dst, src, reinterpret_cast<__ubuf__ int8_t*>(copy_ub), copy_ub_size,
        elem_size, pe, copy_event_id, enable_L2);
}

/**
 * @brief Asynchronous interface. Copy contiguous data on local UB to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param enable_L2         [in] whether to enable L2 cache
 */
SHMEM_DEVICE void shmemx_mte_put_mem_nbi(__gm__ int8_t* dst, __gm__ int8_t* src, uint32_t elem_size, int32_t pe,
    bool enable_L2)
{
        /* Global State Get */
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
        /* CopyUB Config Set */
        uint64_t copy_ub = device_state->mte_config.shmem_ub;
        uint32_t copy_ub_size = device_state->mte_config.ub_size;
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;
        shmemx_mte_put_mem_nbi_low_level(dst, src, reinterpret_cast<__ubuf__ int8_t*>(copy_ub), copy_ub_size,
            elem_size, pe, copy_event_id, enable_L2);
}

#endif