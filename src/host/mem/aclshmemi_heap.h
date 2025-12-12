/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
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

#include "host_device/common_types.h"
#include "utils/aclshmemi_host_types.h"
#include "init/bootstrap/aclshmemi_bootstrap.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"

const int IPC_NAME_SIZE = 65;

class shmem_symmetric_heap {
public:
    shmem_symmetric_heap() {}
    shmem_symmetric_heap(int pe_id, int pe_size, int dev_id);
    ~shmem_symmetric_heap() {};

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
    char names[SHMEM_MAX_RANKS][IPC_NAME_SIZE];

    // pid set to memory_name
    int32_t my_pid = 0UL;
    std::vector<int32_t> pid_list = {};

    // sdid set to memory_name in 910_93
    int64_t my_sdid = 0UL;
    std::vector<int64_t> sdid_list = {};
};


#endif  // SHMEMI_HEAP_H