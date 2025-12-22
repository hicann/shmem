/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <acl/acl.h>
#include "host/aclshmem_host_def.h"
#include "rdma/rdma_manager.h"
#include "utils/aclshmemi_host_types.h"
#include "utils/aclshmemi_logger.h"
#include "host_device/aclshmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif
static rdma_manager* manager;

int aclshmemi_rdma_can_access_peer(int *access, aclshmemi_transport_pe_info_t *peer_info, aclshmemi_transport_pe_info_t *my_info, aclshmemi_transport *t) {
    if (peer_info->pe == my_info->pe) {
        *access = 0;
    } else {
        *access = 1;
    }
    return 0;
}

int aclshmemi_rdma_connect_peers(aclshmemi_transport *t, int *selected_dev_ids, int num_selected_devs, aclshmem_device_host_state_t *state) {
    auto local_device_ip = manager->GetDeviceIP();
    SHM_LOG_INFO("local ip = " << inet_ntoa(local_device_ip));
    std::vector<in_addr> device_ips(state->npes);
    ACLSHMEM_CHECK_RET((g_boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", ACLSHMEM_BOOTSTRAP_ERROR);
    g_boot_handle.allgather(&local_device_ip, device_ips.data(), sizeof(in_addr), &g_boot_handle);
    g_boot_handle.barrier(&g_boot_handle);
    for (int i = 0; i < state->npes; i++) {
        SHM_LOG_INFO("get rank " << i << ", device ip = " << inet_ntoa(device_ips[i]));
    }

    auto local_mr = manager->GetLocalMR();
    SHM_LOG_INFO("local mr = " << local_mr);
    std::vector<RegMemResult> mrs(state->npes);
    g_boot_handle.allgather(&local_mr, mrs.data(), sizeof(RegMemResult), &g_boot_handle);
    for (int i = 0; i < state->npes; i++) {
        state->rdma_device_heap_base[i] = reinterpret_cast<void*>(mrs[i].address);
        SHM_LOG_INFO("get rank " << i << ", mr info = " << mrs[i]);
    }

    HybmTransPrepareOptions TransPrepareOp;
    for (int i = 0; i < state->npes; i++) {
        TransPrepareOp.options[i].nic = std::string(inet_ntoa(device_ips[i])) + ":4647";
        TransPrepareOp.options[i].mr = mrs[i];
    }
    manager->Prepare(TransPrepareOp);
    manager->Connect();
    state->qp_info = reinterpret_cast<uint64_t>(manager->GetQPInfoAddr());
    return 0;
}

int aclshmemi_rdma_finalize(aclshmemi_transport *t, aclshmem_device_host_state_t *state) {
    delete manager;
    return 0;
}

int aclshmemi_rdma_init(aclshmemi_transport *t, aclshmem_device_host_state_t *state) {
    manager = new rdma_manager;
    TransportOptions options;
    options.rankId = state->mype;
    options.rankCount = state->npes;
    options.protocol = 7;
    options.nic = 10002;
    options.dev_id = t->dev_id;
    options.logic_dev_id = t->logical_dev_id;
    manager->OpenDevice(options);

    TransportMemoryRegion mr;
    mr.addr = reinterpret_cast<uint64_t>(state->heap_base);
    mr.size = reinterpret_cast<uint64_t>(state->heap_size);
    manager->RegisterMemoryRegion(mr);
    t->can_access_peer = aclshmemi_rdma_can_access_peer;
    t->connect_peers = aclshmemi_rdma_connect_peers;
    t->finalize = aclshmemi_rdma_finalize;
    return 0;
}

#ifdef __cplusplus
}
#endif