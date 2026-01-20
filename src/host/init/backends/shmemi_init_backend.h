/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_INIT_BACKEND_H
#define SHMEMI_INIT_BACKEND_H

#include <iostream>

#include "host/shmem_host_def.h"
#include "host_device/shmem_common_types.h"

#include "mem/shmemi_heap.h"

#include "init/bootstrap/shmemi_bootstrap.h"

#include "store_op.h"
#include "store_net_group_engine.h"
#include "hybm/include/hybm_def.h"

class aclshmemi_init_backend {
public:
    aclshmemi_init_backend(aclshmemx_init_attr_t *attr, aclshmem_device_host_state_t *global_state, aclshmemx_bootstrap_t bootstrap_flags,
                           aclshmemi_bootstrap_handle_t *handle);
    ~aclshmemi_init_backend();

    int init_device_state();
    int finalize_device_state();
    int update_device_state(void* host_ptr, size_t size);

    int reserve_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int setup_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int remove_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int release_heap(aclshmem_mem_type_t mem_type = DEVICE_SIDE);

    void aclshmemi_global_exit(int status);
    int aclshmemi_control_barrier_all();

private:
    int reach_info_init(void *&gva);
    int create_entity(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int exchange_slice(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int exchange_entity(aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    int32_t init_config_store();
    int32_t init_group_engine();

private:
    shmem_init_attr_t *attributes;
    int npes;
    int device_id;
    bool inited_ = false;
    std::string ip_;
    uint16_t port_ = 9980L;
    hybm_options options_{};
    aclshmem_device_host_state_t *host_state_ = nullptr;
    aclshmem_device_host_state_t *device_state_ = nullptr;
    aclshmemx_bootstrap_t bootstrap_flags_;

    hybm_entity_t hbm_entity_ = nullptr;
    hybm_entity_t dram_entity_ = nullptr;
    shm::store::SmemGroupEnginePtr group_engine_ = nullptr;
    aclshmemi_bootstrap_handle_t *boot_handle_ = nullptr;
    shm::store::StorePtr store_ = nullptr;
    
    void *hbm_gva_ = nullptr;
    void *dram_gva_ = nullptr;
    hybm_mem_slice_t hbm_slice_ = nullptr;
    hybm_mem_slice_t dram_slice_ = nullptr;
};

#endif // SHMEMI_INIT_BACKEND_H