/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEMI_GLOBAL_STATE_H
#define ACLSHMEMI_GLOBAL_STATE_H

#include <iostream>
#include <stdio.h>

#include <acl/acl.h>

#include "host_device/aclshmem_common_types.h"

typedef struct drv_mem_handle drv_mem_handle_t;

struct drv_mem_prop {
    uint32_t side;
    uint32_t devid;
    uint32_t module_id;

    uint32_t pg_type;
    uint32_t mem_type;
    uint64_t reserve;
};

class global_state_reigister {
public:
    global_state_reigister();
    global_state_reigister(int device_id);

    ~global_state_reigister();

    void *get_ptr();
    int get_init_status();
private:
    void *device_ptr_ = nullptr;

    drv_mem_handle_t *alloc_handle;

    int device_id_;

    // 1 means no-init
    int init_status_ = 1;
};


#endif  // ACLSHMEMI_GLOBAL_STATE_H