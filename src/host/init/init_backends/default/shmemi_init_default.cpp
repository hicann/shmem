/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_init_default.h"
#include "utils/shmemi_logger.h"
#include "mem/shmemi_heap_factory.h"

aclshmemi_init_default::aclshmemi_init_default(aclshmemx_init_attr_t *attr, aclshmem_device_host_state_t *global_state)
{
    mype = attr->my_pe;
    npes = attr->n_pes;
    option_attr_ = attr->option_attr;
    host_state_ = global_state;

    auto status = aclrtGetDevice(&device_id);
    if (status != 0) {
        SHM_LOG_ERROR("Get Device_id error");
    }
}

aclshmemi_init_default::~aclshmemi_init_default()
{}

int aclshmemi_init_default::init_device_state()
{
    global_state_d = new global_state_reigister(device_id);
    device_state_ = (aclshmem_device_host_state_t *)std::calloc(1, sizeof(aclshmem_device_host_state_t));
    if (global_state_d->get_init_status() != 0) {
        SHM_LOG_ERROR("global_state reigister error");
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::finalize_device_state()
{
    delete global_state_d;
    std::free(device_state_);
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::update_device_state(void* host_ptr, size_t size)
{
    int32_t ptr_size = npes * sizeof(void *);
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->p2p_device_heap_base, ptr_size, host_state_->p2p_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->rdma_device_heap_base, ptr_size, host_state_->rdma_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->sdma_device_heap_base, ptr_size, host_state_->sdma_device_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    if (host_state_->host_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->p2p_host_heap_base, ptr_size,
            host_state_->p2p_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->rdma_host_heap_base, ptr_size,
            host_state_->rdma_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
        ACLSHMEM_CHECK_RET(aclrtMemcpy(device_state_->sdma_host_heap_base, ptr_size,
            host_state_->sdma_host_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto backup_p2p = device_state_->p2p_device_heap_base;
    auto backup_rdma = device_state_->rdma_device_heap_base;
    auto backup_sdma = device_state_->sdma_device_heap_base;
    auto backup_host_p2p = device_state_->p2p_host_heap_base;
    auto backup_host_rdma = device_state_->rdma_host_heap_base;
    auto backup_host_sdma = device_state_->sdma_host_heap_base;
    std::memcpy(device_state_, host_ptr, size);
    device_state_->p2p_device_heap_base = backup_p2p;
    device_state_->rdma_device_heap_base = backup_rdma;
    device_state_->sdma_device_heap_base = backup_sdma;
    device_state_->p2p_host_heap_base = backup_host_p2p;
    device_state_->rdma_host_heap_base = backup_host_rdma;
    device_state_->sdma_host_heap_base = backup_host_sdma;
    ACLSHMEM_CHECK_RET(aclrtMemcpy(global_state_d->get_ptr(), size, device_state_, size, ACL_MEMCPY_HOST_TO_DEVICE));
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::reserve_heap(aclshmem_mem_type_t mem_type)
{
    if (mem_type == DEVICE_SIDE) {
        heap_obj = aclshmem_symmetric_heap_factory::get_instance().create_heap(mem_type, mype, npes, device_id);
        ACLSHMEM_CHECK_RET(heap_obj->reserve_heap(host_state_->heap_size));
        host_state_->heap_base = heap_obj->get_heap_base();
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->p2p_device_heap_base, npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->rdma_device_heap_base, npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->sdma_device_heap_base, npes * sizeof(void *)));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->p2p_device_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->rdma_device_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->sdma_device_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
        return ACLSHMEM_SUCCESS;
    }
    host_heap_obj = aclshmem_symmetric_heap_factory::get_instance().create_heap(mem_type, mype, npes, device_id);
    ACLSHMEM_CHECK_RET(host_heap_obj->reserve_heap(host_state_->heap_size));
    host_state_->host_heap_base = host_heap_obj->get_heap_base();
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->p2p_host_heap_base, npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->rdma_host_heap_base, npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&host_state_->sdma_host_heap_base, npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->p2p_host_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->rdma_host_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&device_state_->sdma_host_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::setup_heap(aclshmem_mem_type_t mem_type)
{
    if (mem_type == DEVICE_SIDE) {
        ACLSHMEM_CHECK_RET(heap_obj->setup_heap());
        for (int32_t i = 0; i < host_state_->npes; i++) {
            host_state_->p2p_device_heap_base[i] = heap_obj->get_peer_heap_base_p2p(i);
        }
        host_state_->is_aclshmem_created = true;
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(host_heap_obj->setup_heap());
    for (int32_t i = 0; i < host_state_->npes; i++) {
        host_state_->p2p_host_heap_base[i] = host_heap_obj->get_peer_heap_base_p2p(i);
    }
    host_state_->is_aclshmem_created = true;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::remove_heap(aclshmem_mem_type_t mem_type)
{
    if (mem_type == DEVICE_SIDE) {
        ACLSHMEM_CHECK_RET(heap_obj->remove_heap());
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(host_heap_obj->remove_heap());
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::release_heap(aclshmem_mem_type_t mem_type)
{
   auto safe_free_host = [](void**& ptr) -> int {
        if (ptr == nullptr) {
            return ACLSHMEM_SUCCESS;
        }
        int ret = aclrtFreeHost(ptr);
        if (ret == 0) {
            ptr = nullptr;
        }
        return ret;
    };
    auto safe_free_device = [](void**& ptr) -> int {
        if (ptr == nullptr) {
            return ACLSHMEM_SUCCESS;
        }
        int ret = aclrtFree(ptr);
        if (ret == 0) {
            ptr = nullptr;
        }
        return ret;
    };
    if (mem_type == HOST_SIDE) {
        ACLSHMEM_CHECK_RET(safe_free_host(host_state_->p2p_host_heap_base));
        ACLSHMEM_CHECK_RET(safe_free_host(host_state_->rdma_host_heap_base));
        ACLSHMEM_CHECK_RET(safe_free_host(host_state_->sdma_host_heap_base));
        ACLSHMEM_CHECK_RET(safe_free_device(device_state_->p2p_host_heap_base));
        ACLSHMEM_CHECK_RET(safe_free_device(device_state_->rdma_host_heap_base));
        ACLSHMEM_CHECK_RET(safe_free_device(device_state_->sdma_host_heap_base));
        ACLSHMEM_CHECK_RET(host_heap_obj->unreserve_heap());
        return ACLSHMEM_SUCCESS;
    }
    ACLSHMEM_CHECK_RET(safe_free_host(host_state_->p2p_device_heap_base));
    ACLSHMEM_CHECK_RET(safe_free_host(host_state_->rdma_device_heap_base));
    ACLSHMEM_CHECK_RET(safe_free_host(host_state_->sdma_device_heap_base));
    ACLSHMEM_CHECK_RET(safe_free_device(device_state_->p2p_device_heap_base));
    ACLSHMEM_CHECK_RET(safe_free_device(device_state_->rdma_device_heap_base));
    ACLSHMEM_CHECK_RET(safe_free_device(device_state_->sdma_device_heap_base));
    ACLSHMEM_CHECK_RET(heap_obj->unreserve_heap());
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::transport_init()
{
    ACLSHMEM_CHECK_RET(aclshmemi_transport_init(host_state_, option_attr_));      // mte init && rdma init
    ACLSHMEM_CHECK_RET(aclshmemi_build_transport_map(host_state_));               // build transport_map
    ACLSHMEM_CHECK_RET(aclshmemi_transport_setup_connections(host_state_));       // connect_endpoints by transpost_map
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_default::transport_finalize()
{
    ACLSHMEM_CHECK_RET(aclshmemi_transport_finalize(host_state_));
    return ACLSHMEM_SUCCESS;
}

void aclshmemi_init_default::aclshmemi_global_exit(int status)
{
}

int aclshmemi_init_default::aclshmemi_control_barrier_all()
{
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_control_barrier_all_default(aclshmemi_bootstrap_handle_t boot_handle)
{
    ACLSHMEM_CHECK_RET((boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", ACLSHMEM_BOOTSTRAP_ERROR);
    return boot_handle.barrier(&boot_handle);
}