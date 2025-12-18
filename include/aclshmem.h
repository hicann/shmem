/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_API_H
#define ACLSHMEM_API_H

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "device/gm2gm/aclshmem_device_amo.h"
#include "device/gm2gm/aclshmem_device_cc.h"
#include "device/gm2gm/aclshmem_device_mo.h"
#include "device/gm2gm/aclshmem_device_p2p_sync.h"
#include "device/gm2gm/aclshmem_device_rma.h"
#include "device/gm2gm/engine/aclshmem_device_mte.h"
#include "device/gm2gm/engine/aclshmem_device_rdma.h"
#include "device/ub2gm/aclshmem_device_rma.h"
#include "device/ub2gm/engine/aclshmem_device_mte.h"
#include "device/aclshmem_def.h"
#include "device/team/aclshmem_device_team.h"
#endif

#include "host/aclshmem_host_def.h"
#include "host/mem/aclshmem_host_heap.h"
#include "host/init/aclshmem_host_init.h"
#include "host/data_plane/aclshmem_host_rma.h"
#include "host/data_plane/aclshmem_host_p2p_sync.h"
#include "host/team/aclshmem_host_team.h"
#include "host/utils/aclshmem_log.h"

#endif // ACLSHMEM_API_H
