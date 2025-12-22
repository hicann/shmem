/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_INIT_HYBM_H
#define SHMEMI_INIT_HYBM_H

#include <iostream>

#include "init/init_backends/aclshmemi_init_base.h"

#include "host/aclshmem_host_def.h"
#include "host_device/aclshmem_common_types.h"

#include "mem/aclshmemi_global_state.h"
#include "mem/aclshmemi_heap.h"

#include "init/bootstrap/aclshmemi_bootstrap.h"

#include "init/transport/aclshmemi_transport.h"

#include "store_op.h"
#include "store_net_group_engine.h"
#include "hybm/include/hybm_def.h"

class aclshmemi_init_hybm: public aclshmemi_init_base {
public:
    aclshmemi_init_hybm(shmem_init_attr_t *attr, char *ipport, aclshmem_device_host_state_t *global_state);
    ~aclshmemi_init_hybm();

    int init_device_state() override;
    int finalize_device_state() override;
    int update_device_state(void* host_ptr, size_t size) override;

    int reserve_heap() override;
    int setup_heap() override;
    int remove_heap() override;
    int release_heap() override;

    int transport_init() override;
    int transport_finalize() override;
    void aclshmemi_global_exit(int status) override;
    int aclshmemi_control_barrier_all() override;

private:
    int reach_info_init(void *&gva);
    int exchange_slice();
    int exchange_entity();

private:
    shmem_init_attr_t *attributes;
    int npes;
    int device_id;
    bool inited_ = false;
    std::string ip_;
    uint16_t port_ = 9980L;
    hybm_options options_{};
    shm::store::StorePtr store_ = nullptr;
    aclshmem_device_host_state_t *host_state_;
    aclshmem_device_host_state_t *device_state_;
    hybm_entity_t entity_ = nullptr;
    void *gva_ = nullptr;
    hybm_mem_slice_t slice_ = nullptr;
    shm::store::SmemGroupEnginePtr globalGroup_ = nullptr;
};

int32_t aclshmemi_control_barrier_all_default(aclshmemi_bootstrap_handle_t boot_handle);
int32_t aclshmemi_get_uniqueid_hybm(aclshmemx_uniqueid_t *uid);
#endif // SHMEMI_INIT_HYBM_H