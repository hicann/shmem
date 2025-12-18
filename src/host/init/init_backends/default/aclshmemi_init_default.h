
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_INIT_NORMAL_H
#define ACLSHMEMI_INIT_NORMAL_H

#include <iostream>

#include "init/init_backends/aclshmemi_init_base.h"

#include "host/aclshmem_host_def.h"
#include "host_device/aclshmem_common_types.h"

#include "mem/aclshmemi_global_state.h"
#include "mem/aclshmemi_heap.h"

#include "init/bootstrap/aclshmemi_bootstrap.h"

#include "init/transport/aclshmemi_transport.h"

class aclshmemi_init_default: public aclshmemi_init_base {
public:
    aclshmemi_init_default(aclshmemx_init_attr_t *attr, aclshmem_device_host_state_t *global_state);
    ~aclshmemi_init_default();

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
    int mype;
    int npes;
    int device_id;
    aclshmem_device_host_state_t *g_state;

    // global_state
    global_state_reigister *global_state_d = nullptr;

    // heap_obj
    aclshmem_symmetric_heap *heap_obj = nullptr;
    aclshmem_init_optional_attr_t option_attr_;
};

int32_t aclshmemi_control_barrier_all_default(aclshmemi_bootstrap_handle_t boot_handle);

#endif // ACLSHMEMI_INIT_NORMAL_H