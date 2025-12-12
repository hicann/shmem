/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_init_default.h"
#include "utils/aclshmemi_logger.h"

shmemi_init_default::shmemi_init_default(shmem_init_attr_t *attr, shmemi_device_host_state_t *global_state)
{
    mype = attr->my_rank;
    npes = attr->n_ranks;
    option_attr_ = attr->option_attr;
    g_state = global_state;

    auto status = aclrtGetDevice(&device_id);
    if (status != 0) {
        SHM_LOG_ERROR("Get Device_id error");
    }
}

shmemi_init_default::~shmemi_init_default()
{}

int shmemi_init_default::init_device_state()
{
    global_state_d = new global_state_reigister(device_id);
    if (global_state_d->get_init_status() != 0) {
        SHM_LOG_ERROR("global_state reigister error");
    }
    return SHMEM_SUCCESS;
}

int shmemi_init_default::finalize_device_state()
{
    delete global_state_d;
    return SHMEM_SUCCESS;
}

int shmemi_init_default::update_device_state(void* host_ptr, size_t size)
{
    int32_t ptr_size = npes * sizeof(void *);
    SHMEM_CHECK_RET(aclrtMemcpy(g_state->device_p2p_heap_base, ptr_size, g_state->host_p2p_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    SHMEM_CHECK_RET(aclrtMemcpy(g_state->device_rdma_heap_base, ptr_size, g_state->host_rdma_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    SHMEM_CHECK_RET(aclrtMemcpy(g_state->device_sdma_heap_base, ptr_size, g_state->host_sdma_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));

    SHMEM_CHECK_RET(aclrtMemcpy(global_state_d->get_ptr(), size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE));
    return SHMEM_SUCCESS;
}

int shmemi_init_default::reserve_heap()
{
    heap_obj = new shmem_symmetric_heap(mype, npes, device_id);

    SHMEM_CHECK_RET(heap_obj->reserve_heap(g_state->heap_size));

    g_state->heap_base = heap_obj->get_heap_base();

    SHMEM_CHECK_RET(aclrtMallocHost((void **)&g_state->host_p2p_heap_base, npes * sizeof(void *)));
    SHMEM_CHECK_RET(aclrtMallocHost((void **)&g_state->host_rdma_heap_base, npes * sizeof(void *)));
    SHMEM_CHECK_RET(aclrtMallocHost((void **)&g_state->host_sdma_heap_base, npes * sizeof(void *)));

    SHMEM_CHECK_RET(aclrtMalloc((void **)&g_state->device_p2p_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    SHMEM_CHECK_RET(aclrtMalloc((void **)&g_state->device_rdma_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    SHMEM_CHECK_RET(aclrtMalloc((void **)&g_state->device_sdma_heap_base, npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    return SHMEM_SUCCESS;
}

int shmemi_init_default::setup_heap()
{
    SHMEM_CHECK_RET(heap_obj->setup_heap());

    for (int32_t i = 0; i < g_state->npes; i++) {
        g_state->host_p2p_heap_base[i] = heap_obj->get_peer_heap_base_p2p(i);
    }
    g_state->is_shmem_created = true;

    return SHMEM_SUCCESS;
}

int shmemi_init_default::remove_heap()
{
    SHMEM_CHECK_RET(heap_obj->remove_heap());
    return SHMEM_SUCCESS;
}

int shmemi_init_default::release_heap()
{
    if (g_state->host_p2p_heap_base != nullptr) {
        SHMEM_CHECK_RET(aclrtFreeHost(g_state->host_p2p_heap_base));
    }
    if (g_state->host_p2p_heap_base != nullptr) {
        SHMEM_CHECK_RET(aclrtFreeHost(g_state->host_rdma_heap_base));
    }
    if (g_state->host_p2p_heap_base != nullptr) {
        SHMEM_CHECK_RET(aclrtFreeHost(g_state->host_sdma_heap_base));
    }
    if (g_state->device_p2p_heap_base != nullptr) {
        SHMEM_CHECK_RET(aclrtFree(g_state->device_p2p_heap_base));
    }
    if (g_state->device_rdma_heap_base != nullptr) {
        SHMEM_CHECK_RET(aclrtFree(g_state->device_rdma_heap_base));
    }
    if (g_state->device_sdma_heap_base != nullptr) {
        SHMEM_CHECK_RET(aclrtFree(g_state->device_sdma_heap_base));
    }
    SHMEM_CHECK_RET(heap_obj->unreserve_heap());
    return SHMEM_SUCCESS;
}

int shmemi_init_default::transport_init()
{
    SHMEM_CHECK_RET(shmemi_transport_init(g_state, option_attr_));      // mte init && rdma init
    SHMEM_CHECK_RET(shmemi_build_transport_map(g_state));               // build transport_map
    SHMEM_CHECK_RET(shmemi_transport_setup_connections(g_state));       // connect_endpoints by transpost_map
    return SHMEM_SUCCESS;
}

int shmemi_init_default::transport_finalize()
{
    SHMEM_CHECK_RET(shmemi_transport_finalize(g_state));
    return SHMEM_SUCCESS;
}

int32_t shmemi_control_barrier_all_default(shmemi_bootstrap_handle_t boot_handle)
{
    SHMEM_CHECK_RET((boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", SHMEM_BOOTSTRAP_ERROR);
    return boot_handle.barrier(&boot_handle);
}