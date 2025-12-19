/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <fstream>
#include "mem/aclshmemi_heap.h"
#include "aclshmemi_host_common.h"
#include "dlfcn.h"

#include "init/transport/aclshmemi_transport.h"

static void *transport_mte_lib = NULL;
static void *transport_rdma_lib = NULL;

uint64_t *host_hash_list;

aclshmemi_host_trans_state_t g_host_state;

int32_t aclshmemi_transport_init(aclshmem_device_host_state_t *g_state, aclshmem_init_optional_attr_t& option_attr) {
    // Initialize MTE by default
    g_host_state.num_choosen_transport = 1;
    g_host_state.transport_map = (int *)calloc(g_state->npes * g_state->npes, sizeof(int));
    g_host_state.pe_info = (aclshmemi_transport_pe_info *)calloc(g_state->npes, sizeof(aclshmemi_transport_pe_info));

    transport_mte_lib = dlopen("aclshmem_transport_mte.so", RTLD_NOW);
    if (!transport_mte_lib) {
        SHM_LOG_ERROR("Transport unable to load " << "aclshmem_transport_mte.so" << ", err is: " << stderr);
        return ACLSHMEM_INVALID_VALUE;
    }

    transport_init_func init_mte_fn;
    init_mte_fn = (transport_init_func)dlsym(transport_mte_lib, "aclshmemi_mte_init");
    if (!init_mte_fn) {
        dlclose(transport_mte_lib);
        transport_mte_lib = NULL;
        SHM_LOG_ERROR("Unable to get info from " << "aclshmem_transport_mte.so" << ".");
        return ACLSHMEM_INVALID_VALUE;
    }

    // Package my_info
    int32_t device_id;
    int64_t server_id;
    int64_t superpod_id;
    ACLSHMEM_CHECK_RET(aclrtGetDevice(&device_id));
    const int infoTypeServerId = 27;
    ACLSHMEM_CHECK_RET(rtGetDeviceInfo(device_id, 0, infoTypeServerId, &server_id));
    const int infoTypeSuperpodId = 29;
    ACLSHMEM_CHECK_RET(rtGetDeviceInfo(device_id, 0, infoTypeSuperpodId, &superpod_id));
        
    aclshmemi_transport_pe_info_t my_info;
    my_info.pe = g_state->mype;
    my_info.dev_id = device_id;
    my_info.server_id = server_id;
    my_info.superpod_id = superpod_id;

    // server_id invalid
    if (server_id == 0x3FFU) {
        static uint32_t bootIdHead;
        static std::string sysBootId;

        std::string bootIdPath("/proc/sys/kernel/random/boot_id");
        std::ifstream input(bootIdPath);
        input >> sysBootId;

        std::stringstream ss(sysBootId);
        ss >> std::hex >> bootIdHead;

        my_info.server_id = bootIdHead;
    }
    
    // AllGather All pe's host info
    ACLSHMEM_CHECK_RET((g_boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", ACLSHMEM_BOOTSTRAP_ERROR);
    g_boot_handle.allgather((void *)&my_info, g_host_state.pe_info, sizeof(aclshmemi_transport_pe_info_t), &g_boot_handle);
    ACLSHMEM_CHECK_RET(init_mte_fn(&g_host_state.choosen_transports[0], g_state));

    // If enable RDMA
    if (option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_ROCE) {
        g_host_state.num_choosen_transport++;

        int32_t logicDeviceId = -1;
        rtLibLoader& loader = rtLibLoader::getInstance();
        if (loader.isLoaded()) {
            loader.getLogicDevId(device_id, &logicDeviceId);
        }
        g_host_state.choosen_transports[1].logical_dev_id = logicDeviceId;
        g_host_state.choosen_transports[1].dev_id = device_id;

        transport_rdma_lib = dlopen("aclshmem_transport_rdma.so", RTLD_NOW);
        if (!transport_rdma_lib) {
            SHM_LOG_ERROR("Transport unable to load " << "aclshmem_transport_rdma.so" << ", err is: " << stderr);
            return ACLSHMEM_INVALID_VALUE;
        }

        transport_init_func init_rdma_fn;
        init_rdma_fn = (transport_init_func)dlsym(transport_rdma_lib, "aclshmemi_rdma_init");
        if (!init_rdma_fn) {
            dlclose(transport_rdma_lib);
            transport_rdma_lib = NULL;
            SHM_LOG_ERROR("Unable to get info from " << "aclshmem_transport_rdma.so" << ".");
            return ACLSHMEM_INVALID_VALUE;
        }
        ACLSHMEM_CHECK_RET(init_rdma_fn(&g_host_state.choosen_transports[1], g_state));
    }

    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_build_transport_map(aclshmem_device_host_state_t *g_state) {
    int *local_map = NULL;
    local_map = (int *)calloc(g_state->npes, sizeof(int));

    aclshmemi_transport_t t;

    // Loop can_access_peer, j = 0 means MTE, j = 1 means RDMA ...
    for (int j = 0; j < g_host_state.num_choosen_transport; j++) {
        t = g_host_state.choosen_transports[j];

        for (int i = 0; i < g_state->npes; i++) {
            int reach = 0;

            ACLSHMEM_CHECK_RET(t.can_access_peer(&reach, g_host_state.pe_info + i, g_host_state.pe_info + g_state->mype, &t));

            if (reach) {
                int m = 1 << j;
                local_map[i] |= m;
            }
        }
    }

    for (int i = 0; i < g_state->npes; i++) {
        g_state->topo_list[i] = static_cast<uint8_t>(local_map[i]);
    }
    ACLSHMEM_CHECK_RET((g_boot_handle.is_bootstraped != true), "boot_handle not bootstraped, Please check if the method call occurs before initialization or after finalization.", ACLSHMEM_BOOTSTRAP_ERROR);
    g_boot_handle.allgather(local_map, g_host_state.transport_map, g_state->npes * sizeof(int), &g_boot_handle);

    if (local_map) free(local_map);
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_transport_setup_connections(aclshmem_device_host_state_t *g_state) {
    aclshmemi_transport_t t;
    // MTE 
    t = g_host_state.choosen_transports[0];

    int *mte_peer_list;
    int mte_peer_num = 0;
    mte_peer_list = (int *)calloc(g_state->npes, sizeof(int));

    int local_offset = g_state->mype * g_state->npes;
    for (int i = 0; i < g_state->npes; i++) {
        if (i == g_state->mype)
            continue;
        /* Check if MTE connected. */
        if (g_host_state.transport_map[local_offset + i] & 0x1) {
            aclshmemi_transport_pe_info_t *peer_info = (g_host_state.pe_info + i);
            aclshmemi_transport_pe_info_t *my_info = (g_host_state.pe_info + g_state->mype);
            // Only PEs in the same Node need to build up MTE connection.
            if (my_info->server_id == peer_info->server_id) {
                mte_peer_list[mte_peer_num] = peer_info->dev_id;
                ++mte_peer_num;
            }
        }
    }

    t.connect_peers(&t, mte_peer_list, mte_peer_num, g_state);

    if (g_host_state.num_choosen_transport > 1) {
        int *rdma_peer_list;
        int rdma_peer_num = 0;
        rdma_peer_list = (int *)calloc(g_state->npes, sizeof(int));

        int local_offset = g_state->mype * g_state->npes;
        for (int i = 0; i < g_state->npes; i++) {
            if (i == g_state->mype)
                continue;
            if (g_host_state.transport_map[local_offset + i] & 2) {
                aclshmemi_transport_pe_info_t *peer_info = (g_host_state.pe_info + i);
                rdma_peer_list[rdma_peer_num] = peer_info->dev_id;
                ++rdma_peer_num;
            }
        }
        t = g_host_state.choosen_transports[1];
        t.connect_peers(&t, rdma_peer_list, rdma_peer_num, g_state);
    }

    return 0;
}

int32_t aclshmemi_transport_finalize(aclshmem_device_host_state_t *g_state) {
    aclshmemi_transport_t t;
    // MTE 
    t = g_host_state.choosen_transports[0];
    t.finalize(&t, g_state);

    if (transport_mte_lib != NULL) {
        dlclose(transport_mte_lib);
        transport_mte_lib = NULL;
    }

    if (g_host_state.num_choosen_transport > 1) {
        t = g_host_state.choosen_transports[1];
        t.finalize(&t, g_state);

        if (transport_rdma_lib != NULL) {
            dlclose(transport_rdma_lib);
            transport_rdma_lib = NULL;
        }
    }
    return 0;
}
