/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_HEAP_H
#define SHMEMI_HEAP_H

#include <iostream>
#include <vector>
#include <map>
#include <mutex>

#include <acl/acl.h>
#include "shmemi_heap_base.h"
#include "host_device/shmem_common_types.h"
#include "utils/shmemi_host_types.h"
#include "init/bootstrap/shmemi_bootstrap.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"

#include "shmemi_heap_factory.h"

const int IPC_NAME_SIZE = 65;

class aclshmem_symmetric_heap : public aclshmem_symmetric_heap_base {
public:
    aclshmem_symmetric_heap() {}
    aclshmem_symmetric_heap(int pe_id, int pe_size, int dev_id, aclshmem_mem_type_t mem_type = DEVICE_SIDE);
    ~aclshmem_symmetric_heap() {};

    int reserve_heap(size_t size);              // aclrtReserveMemAddress && aclrtMallocPhysical
    int unreserve_heap();                       // halMemAddressFree && aclrtFreePhysical

    int setup_heap();                           // export && import p2p memories && aclrtMapMem
    int remove_heap();                          // aclrtUnmapMem

    void *get_heap_base();                      // return heap_base_
    void *get_peer_heap_base_p2p(int pe_id);    // peer_heap_base_p2p_

private:
    int export_memory();
    int import_memory();

    int export_pid();
    int import_pid();

    int32_t mype;
    int32_t npes;
    int32_t device_id;

    uint64_t alloc_size;

    void *heap_base_;
    void **peer_heap_base_p2p_;

    // handle used to map local virtual ptr
    aclrtPhysicalMemProp memprop;
    aclrtDrvMemHandle local_handle;

    // names used to share memory
    std::string memory_name;
    char names[ACLSHMEM_MAX_PES][IPC_NAME_SIZE];

    // pid set to memory_name
    int32_t my_pid = 0UL;
    std::vector<int32_t> pid_list = {};

    // sdid set to memory_name in 910_93
    int64_t my_sdid = 0UL;
    std::vector<int64_t> sdid_list = {};
};

// 注册DEVICE_SIDE堆实现
REGISTER_HEAP_CREATOR(DEVICE_SIDE, aclshmem_symmetric_heap);

#endif  // ACLSHMEMI_HEAP_H