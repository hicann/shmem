/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_INIT_BASE_H
#define SHMEMI_INIT_BASE_H

#include <iostream>

#include "acl/acl.h"
#include "host_device/common_types.h"

class shmemi_init_base {
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

    virtual ~shmemi_init_base() {}
};

#endif // SHMEMI_INIT_BASE_H