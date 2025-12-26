/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shmemi_heap_shm.h"

#include "acl/acl.h"
#include "utils/shmemi_logger.h"
#include "./generic_guard.h"
#include "init/bootstrap/shmemi_bootstrap.h"

aclshmemi_symmetric_heap::aclshmemi_symmetric_heap(int rank_id, int rank_size,
    int device_id, aclshmem_mem_type_t mem_type) :
    rank_id_(rank_id),
    rank_size_(rank_size),
    device_id_(device_id),
    mem_type_(mem_type)
{}

int32_t aclshmemi_symmetric_heap::set_mem_access()
{
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (mem_type_ == HOST_SIDE) {
        aclrtMemLocation loc;
        loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        loc.id = rank_id_;
        aclrtMemAccessDesc des;
        des.location = loc;
        des.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        ACLSHMEM_CHECK_RET(aclrtMemSetAccess(peer_heap_base_p2p_[rank_id_], heap_size_, &des, 1),
            "aclrtMemSetAccess failed.", ACLSHMEM_SMEM_ERROR);
    }
#endif
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::malloc_physical_memory()
{
    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = (mem_type_ == HOST_SIDE) ? 0 : device_id_;
    prop.location.type = (mem_type_ == HOST_SIDE) ? ACL_MEM_LOCATION_TYPE_HOST : ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;
    ACLSHMEM_CHECK_RET(aclrtMallocPhysical(&local_handle_, heap_size_, &prop, 0),
        "aclrtMallocPhysical failed", ACLSHMEM_SMEM_ERROR);
    auto local_handle_guard = shmem::make_guard(local_handle_, [](aclrtDrvMemHandle p) {
        if (p != nullptr) {
            (void)aclrtFreePhysical(p);
        }
    });
    ACLSHMEM_CHECK_RET(aclrtMapMem(peer_heap_base_p2p_[rank_id_], heap_size_, 0, local_handle_, 0),
        "aclrtMapMem failed", ACLSHMEM_SMEM_ERROR);
    auto peer_map_guard = shmem::make_guard(peer_heap_base_p2p_[rank_id_], [](void *p) {
        if (p != nullptr) {
            (void)aclrtUnmapMem(p);
        }
    });
    if (set_mem_access() != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_SMEM_ERROR;
    }
    peer_map_guard.release_ownership();
    local_handle_guard.release_ownership();
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::reserve_memory_addr(void **va)
{
    for (size_t i = 0; i < rank_size_; ++i) {
        ACLSHMEM_CHECK_RET(aclrtReserveMemAddress(&peer_heap_base_p2p_[i], heap_size_, 0, nullptr, 1),
            "aclrtReserveMemAddress failed", ACLSHMEM_SMEM_ERROR);
    }
    heap_base_ = peer_heap_base_p2p_[rank_id_];
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::reserve_heap(size_t size)
{
    peer_heap_base_p2p_ = (void **)std::calloc(rank_size_, sizeof(void *));
    heap_size_ = size;
    for (int i = 0; i < rank_size_; i++) {
        peer_heap_base_p2p_[i] = nullptr;
    }
    auto peer_guard = shmem::make_guard(peer_heap_base_p2p_, [](void **p) {
        if (p != nullptr) {
            std::free(p);
            p = nullptr;
        }
    });
    void *va = nullptr;
    if (reserve_memory_addr(&va) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_SMEM_ERROR;
    }
    peer_guard.release_ownership();
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::setup_heap()
{
    ACLSHMEM_CHECK_RET(malloc_physical_memory());
    ACLSHMEM_CHECK_RET(export_memory());
    ACLSHMEM_CHECK_RET(import_memory());
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::export_memory()
{
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    auto handle_type = mem_type_ == HOST_SIDE ? ACL_MEM_SHARE_HANDLE_TYPE_FABRIC : ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
    ACLSHMEM_CHECK_RET(aclrtMemExportToShareableHandleV2(local_handle_, 1, handle_type, &shareable_handle_),
        "aclrtMemExportToShareableHandleV2 failed", ACLSHMEM_SMEM_ERROR);
#endif
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::import_memory()
{
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    void* send_buf = &shareable_handle_;
    void* recv_buf = nullptr;
    size_t buf_size = rank_size_ * sizeof(shareable_handle_);
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void**)&recv_buf, buf_size), "aclrtMallocHost failed",
        ACLSHMEM_SMEM_ERROR);
    g_boot_handle.allgather(send_buf, recv_buf, sizeof(shareable_handle_), &g_boot_handle);
    auto phy_array = static_cast<aclrtMemFabricHandle*>(recv_buf);
    for (uint32_t i = 0; i < rank_size_; i++) {
        if (i == rank_id_) {
            continue;
        }
        aclrtDrvMemHandle imported_handle;
        auto handle_type = mem_type_ == HOST_SIDE ? ACL_MEM_SHARE_HANDLE_TYPE_FABRIC : ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
        ACLSHMEM_CHECK_RET(aclrtMemImportFromShareableHandleV2(&phy_array[i], handle_type, 0, &imported_handle),
            "aclrtMemImportFromShareableHandleV2 failed", ACLSHMEM_SMEM_ERROR);
        ACLSHMEM_CHECK_RET(aclrtMapMem(peer_heap_base_p2p_[i], heap_size_, 0, imported_handle, 0),
            "aclrtMapMem remote mem failed", ACLSHMEM_SMEM_ERROR);
    }
    g_boot_handle.barrier(&g_boot_handle);
#endif
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::unmap()
{
    if (local_handle_ != nullptr) {
        return ACLSHMEM_SUCCESS;
    }
    // 取消映射所有进程的内存
    for (uint32_t i = 0; i < rank_size_; i++) {
        void *unmap_addr = peer_heap_base_p2p_[i];
        if (unmap_addr == nullptr) {
            continue;
        }
        ACLSHMEM_CHECK_RET(aclrtUnmapMem(unmap_addr),
            "Failed to unmap memory", ACLSHMEM_SMEM_ERROR);
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::release_memory()
{
    void* base_va = peer_heap_base_p2p_[0];
    for (uint32_t i = 0; i < rank_size_; i++) {
        void *release_addr = peer_heap_base_p2p_[i];
        if (release_addr != nullptr) {
            aclrtReleaseMemAddress(reinterpret_cast<char *>(release_addr));
        }
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_symmetric_heap::unreserve_heap()
{
    ACLSHMEM_CHECK_RET(release_memory());
    if (local_handle_ != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreePhysical(local_handle_));
    }
    if (peer_heap_base_p2p_ != nullptr) {
        std::free(peer_heap_base_p2p_);
        peer_heap_base_p2p_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_symmetric_heap::remove_heap()
{
    if (unmap() != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_SMEM_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}