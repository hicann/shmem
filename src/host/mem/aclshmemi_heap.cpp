/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_heap.h"
#include "host/aclshmem_host_def.h"
#include "utils/aclshmemi_host_types.h"
#include "utils/aclshmemi_logger.h"

shmem_symmetric_heap::shmem_symmetric_heap(int pe_id, int pe_size, int dev_id): mype(pe_id), npes(pe_size), device_id(dev_id)
{
    pid_list.resize(pe_size);
    sdid_list.resize(pe_size);

    memprop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    memprop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    memprop.memAttr = ACL_HBM_MEM_HUGE;
    memprop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    memprop.location.id = dev_id;
    memprop.reserve = 0;
}

int shmem_symmetric_heap::reserve_heap(size_t size)
{
    peer_heap_base_p2p_ = (void **)std::calloc(npes, sizeof(void *));

    // reserve virtual ptrs
    for (int i = 0; i < npes; i++) {
        peer_heap_base_p2p_[i] = NULL;
    }

    // reserve local heap_base_
    SHMEM_CHECK_RET(aclrtReserveMemAddress(&(peer_heap_base_p2p_[mype]), size, 0, nullptr, 1));
    heap_base_ = peer_heap_base_p2p_[mype];

    // alloc local physical memory
    SHMEM_CHECK_RET(aclrtMallocPhysical(&local_handle, size, &memprop, 0));

    alloc_size = size;
    SHMEM_CHECK_RET(aclrtMapMem(peer_heap_base_p2p_[mype], alloc_size, 0, local_handle, 0));

    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::export_memory()
{
    // Get memory_name
    char memoryName[IPC_NAME_SIZE] = {};
    SHMEM_CHECK_RET(rtIpcSetMemoryName(peer_heap_base_p2p_[mype], alloc_size, memoryName, IPC_NAME_SIZE));

    memory_name = memoryName;
    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::export_pid()
{
    // Get local pid
    SHMEM_CHECK_RET(aclrtDeviceGetBareTgid(&my_pid));

    // Get Sdid
    const int rtModuleTypeSystem = 0;
    const int infoTypeSdid = 26;
    SHMEM_CHECK_RET(rtGetDeviceInfo(device_id, rtModuleTypeSystem, infoTypeSdid, &my_sdid));

    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::import_pid()
{
    // Get all pids
    SHMEM_CHECK_RET((g_boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", SHMEM_BOOTSTRAP_ERROR);
    g_boot_handle.allgather(&my_pid, pid_list.data(), 1 * sizeof(int), &g_boot_handle);

    // Get all sdids
    g_boot_handle.allgather(&my_sdid, sdid_list.data(), 1 * sizeof(int64_t), &g_boot_handle);

    // Set Sdid and pid into Shared Memory
    int local_offset = mype * npes;
    for (int i = 0; i < npes; i++) {
        if (i == mype || !(g_host_state.transport_map[local_offset + i] & 0x1)) {
            continue;
        }
        SHMEM_CHECK_RET(rtSetIpcMemorySuperPodPid(memory_name.c_str(), sdid_list[i], &pid_list[i], 1));
    }

    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::import_memory()
{
    SHMEM_CHECK_RET((g_boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", SHMEM_BOOTSTRAP_ERROR);
    g_boot_handle.allgather(memory_name.c_str(), names, IPC_NAME_SIZE, &g_boot_handle);

    static std::mutex mut;
    std::lock_guard<std::mutex> lock(mut);

    int local_offset = mype * npes;
    for (int i = 0; i < npes; i++) {
        if (i == mype || !(g_host_state.transport_map[local_offset + i] & 0x1)) {
            continue;
        }
        SHMEM_CHECK_RET(rtIpcOpenMemory(reinterpret_cast<void **>(&peer_heap_base_p2p_[i]), names[i]));
    }

    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::setup_heap()
{
    SHMEM_CHECK_RET(export_memory());
    SHMEM_CHECK_RET(export_pid());
    SHMEM_CHECK_RET(import_pid());
    SHMEM_CHECK_RET(import_memory());

    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::remove_heap()
{
    for (int i = 0; i < npes; i++) {
        if (i == mype || peer_heap_base_p2p_[i] == NULL) {
            continue;
        }
        SHMEM_CHECK_RET(rtIpcCloseMemory(static_cast<void *>(peer_heap_base_p2p_[i])));
        peer_heap_base_p2p_[i] = NULL;
    }

    // This barrier is necessary, otherwise Unmap will fail.
    SHMEM_CHECK_RET((g_boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", SHMEM_BOOTSTRAP_ERROR);
    g_boot_handle.barrier(&g_boot_handle);

    SHMEM_CHECK_RET(rtIpcDestroyMemoryName(memory_name.c_str()));

    SHMEM_CHECK_RET(aclrtUnmapMem(peer_heap_base_p2p_[mype]));
    return SHMEM_SUCCESS;
}

int shmem_symmetric_heap::unreserve_heap()
{
    for (int i = 0; i < npes; i++) {
        if (peer_heap_base_p2p_[i] != NULL) {
            SHMEM_CHECK_RET(aclrtReleaseMemAddress(peer_heap_base_p2p_[i]));
        }
    }
    SHMEM_CHECK_RET(aclrtFreePhysical(local_handle));
    return SHMEM_SUCCESS;
}

void *shmem_symmetric_heap::get_heap_base()
{
    return heap_base_;
}

void *shmem_symmetric_heap::get_peer_heap_base_p2p(int pe_id)
{
    return peer_heap_base_p2p_[pe_id];
}