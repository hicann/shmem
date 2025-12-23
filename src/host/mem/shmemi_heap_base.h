/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _HOST_MEM_ACLSHMEMI_SYMMETRIC_HEAP_BASE_H_
#define _HOST_MEM_ACLSHMEMI_SYMMETRIC_HEAP_BASE_H_

#include <iostream>
#include <vector>
#include <map>
#include <mutex>
#include "host_device/shmem_common_types.h"
#include "utils/shmemi_host_types.h"

class aclshmem_symmetric_heap_base {
public:
    aclshmem_symmetric_heap_base()= default;
    aclshmem_symmetric_heap_base(int pe_id, int pe_size, int dev_id, aclshmem_mem_type_t mem_type = DEVICE_SIDE) {}
    virtual ~aclshmem_symmetric_heap_base() = default;
    virtual int reserve_heap(size_t size) = 0;              // aclrtReserveMemAddress && aclrtMallocPhysical
    virtual int unreserve_heap() = 0;                       // halMemAddressFree && aclrtFreePhysical
    virtual int setup_heap() = 0;                           // export && import p2p memories && aclrtMapMem
    virtual int remove_heap() = 0;                          // aclrtUnmapMem
    virtual void *get_heap_base() = 0;                      // return heap_base_
    virtual void *get_peer_heap_base_p2p(int pe_id) = 0;    // peer_heap_base_p2p_
};
#endif  // _HOST_MEM_ACLSHMEMI_SYMMETRIC_HEAP_BASE_H_