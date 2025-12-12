/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_TRANSPORT_H
#define SHMEMI_TRANSPORT_H

typedef int(*transport_init_func)(shmemi_transport_t *transport, shmemi_device_host_state_t *g_state);

int32_t shmemi_transport_init(shmemi_device_host_state_t *g_state, shmem_init_optional_attr_t &option_attr);

int32_t shmemi_build_transport_map(shmemi_device_host_state_t *g_state);

int32_t shmemi_transport_setup_connections(shmemi_device_host_state_t *g_state);

int32_t shmemi_transport_finalize(shmemi_device_host_state_t *g_state);

#endif  // SHMEMI_TRANSPORT_H