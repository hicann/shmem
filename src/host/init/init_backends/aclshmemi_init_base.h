/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_INIT_BASE_H
#define ACLSHMEMI_INIT_BASE_H

#include <iostream>

#include "acl/acl.h"
#include "host_device/aclshmem_common_types.h"

class aclshmemi_init_base {
public:
    virtual int init_device_state() = 0;
    virtual int finalize_device_state() = 0;
    virtual int update_device_state(void* host_ptr, size_t size) = 0;

    virtual int reserve_heap() = 0;
    virtual int setup_heap() = 0;
    virtual int remove_heap() = 0;
    virtual int release_heap() = 0;

    virtual int transport_init() = 0;
    virtual int transport_finalize() = 0;
    virtual void aclshmemi_global_exit(int status) = 0;
    virtual int aclshmemi_control_barrier_all() = 0;

    virtual ~aclshmemi_init_base() {}
};

#endif // ACLSHMEMI_INIT_BASE_H