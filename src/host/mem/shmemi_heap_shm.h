/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HOST_MEM_SHMEMI_HEAP_SHM_H
#define HOST_MEM_SHMEMI_HEAP_SHM_H
#include "host_device/shmem_common_types.h"
#include "shmemi_heap_base.h"
#include "acl/acl.h"
#include "shmemi_heap_factory.h"

class aclshmemi_symmetric_heap : public aclshmem_symmetric_heap_base {
public:
    aclshmemi_symmetric_heap(int rank_id, int rank_size, int device_id, aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    ~aclshmemi_symmetric_heap() = default;
    int32_t reserve_heap(size_t size);
    int32_t unreserve_heap();
    int32_t setup_heap();
    int32_t remove_heap();
    void *get_heap_base() {
        return heap_base_;
    }
    size_t get_heap_size() {
        return heap_size_;
    }
    void *get_peer_heap_base_p2p(int rank_id) {
        if (peer_heap_base_p2p_ == nullptr) {
            return nullptr;
        }
        return peer_heap_base_p2p_[rank_id];
    }
protected:
    int32_t malloc_physical_memory();
    int32_t set_mem_access();
    int32_t reserve_memory_addr(void **va);
    int32_t release_memory();
    int32_t export_memory();
    int32_t import_memory();
    int32_t unmap();
    size_t rank_id_;
    size_t rank_size_;
    int32_t device_id_;
    size_t heap_size_ = 0;
    aclshmem_mem_type_t mem_type_ = DEVICE_SIDE;
    void *heap_base_ = nullptr;
    void **peer_heap_base_p2p_ = nullptr;
    aclrtDrvMemHandle local_handle_;
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    aclrtMemFabricHandle shareable_handle_;
#endif
};
REGISTER_HEAP_CREATOR(HOST_SIDE, aclshmemi_symmetric_heap);

#endif
