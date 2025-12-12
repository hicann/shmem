/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <acl/acl.h>
#include "mem/aclshmemi_heap.h"
#include "aclshmemi_host_common.h"
#include "host_device/common_types.h"
#include "init/transport/shmemi_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

int shmemi_mte_can_access_peer(int *access, shmemi_transport_pe_info_t *peer_info, shmemi_transport_pe_info_t *my_info, shmemi_transport *t) {
    // origin access set to 0.
    *access = 0;

    auto sName = aclrtGetSocName();
    std::string socName{sName};
    if (socName.find("Ascend910B") != std::string::npos) {                  // Ascend910B Topo
        // Check Server ID.
        if (my_info->server_id != peer_info->server_id) {
            *access = 0;
            return 0;
        }

        // In same node, Check HCCS Connectivity.
        int64_t hccs_connected = -1;
        SHMEM_CHECK_RET(rtGetPairDevicesInfo(my_info->pe, peer_info->dev_id, 0, &hccs_connected));

        // In 910B, Flag 0 -> HCCS.
        const static int SELF_FLAG = 0;
        if (hccs_connected == SELF_FLAG) {
            *access = 1;
        }
    } else if (socName.find("Ascend910_93") != std::string::npos) {         // Ascend910_93 Topo
        // In same node, Check HCCS Connectivity.
        int64_t hccs_connected = -1;
        /* TODO: This func now doesn't support 910_93 multiNode HCCS Check. Only Check in the same Node. */
        SHMEM_CHECK_RET(rtGetPairDevicesInfo(my_info->pe, peer_info->dev_id, 0, &hccs_connected));

        // In 910_93, Flag 0 -> SELF, 5 -> SIO, 6 -> HCCS.
        const static int SELF_FLAG = 0;
        const static int SIO_FLAG = 5;
        const static int HCCS_FLAG = 6;
        if (hccs_connected == SELF_FLAG || hccs_connected == SIO_FLAG || hccs_connected == HCCS_FLAG) {
            *access = 1;
        }
    }

    return 0;
}

int shmemi_mte_connect_peers(shmemi_transport *t, int *selected_dev_ids, int num_selected_devs, shmemi_device_host_state_t *g_state) {
    // EnablePeerAccess
    for (int i = 0; i < num_selected_devs; i++) {
        SHMEM_CHECK_RET(aclrtDeviceEnablePeerAccess(selected_dev_ids[i], 0));
    }
    return 0;
}

int shmemi_mte_finalize(shmemi_transport *t, shmemi_device_host_state_t *g_state) {
    return 0;
}

// control plane
int shmemi_mte_init(shmemi_transport_t *t, shmemi_device_host_state_t *g_state) {
    t->can_access_peer = shmemi_mte_can_access_peer;
    t->connect_peers = shmemi_mte_connect_peers;
    t->finalize = shmemi_mte_finalize;

    return 0;
}

#ifdef __cplusplus
}
#endif