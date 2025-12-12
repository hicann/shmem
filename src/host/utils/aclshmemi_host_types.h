/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_HOST_TYPES_H
#define SHMEMI_HOST_TYPES_H

#define SHMEM_MAX_TRANSPORT_NUM 16

#include "host_device/common_types.h"
#define SHMEM_MAX_HANDLE_IP_PORT_LEN 64
typedef struct shmemi_bootstrap_attr {
    shmemi_bootstrap_attr() : initialize_mf(0), mpi_comm(NULL), uid_args(NULL)
    {}
    int initialize_mf;
    void *mpi_comm;
    void *mete_data;
    void *uid_args;
} shmemi_bootstrap_attr_t;

typedef struct shmemi_bootstrap_init_ops {
    void *cookie;
    int (*get_unique_id)(void *cookit);
    int (*get_unique_id_static_magic)(void *uid_info, bool is_root);
} shmemi_bootstrap_init_ops_t;

typedef struct shmemi_bootstrap_handle {
    int32_t     mype, npes;
    void        *bootstrap_state;

    int (*finalize)(shmemi_bootstrap_handle *boot_handle);
    int (*allgather)(const void *sendbuf, void *recvbuf, int size, shmemi_bootstrap_handle *boot_handle);
    int (*barrier)(shmemi_bootstrap_handle *boot_handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int size, shmemi_bootstrap_handle *boot_handle);
    void (*global_exit)(int status);
    shmemi_bootstrap_init_ops_t *pre_init_ops;
    bool is_bootstraped = false;
    char ipport[SHMEM_MAX_HANDLE_IP_PORT_LEN];
    bool use_attr_ipport = false;
} shmemi_bootstrap_handle_t;

typedef struct shmemi_bootstrap_mpi_options {
    // TBD
} shmemi_bootstrap_mpi_options_t;

typedef struct shmemi_bootstrap_uid_options {
    // TBD
} shmemi_bootstrap_uid_options_t;

typedef struct shmemi_transport_pe_info {
    int32_t     pe;
    int32_t     dev_id;
    int64_t     server_id;
    int64_t     superpod_id;
} shmemi_transport_pe_info_t;

typedef struct shmemi_transport {
    // control plane
    int (*can_access_peer)(int *access, shmemi_transport_pe_info_t *peer_info,
                            shmemi_transport_pe_info_t *my_info, struct shmemi_transport *t);
    int (*connect_peers)(struct shmemi_transport *t, int *selected_dev_ids,
                            int num_selected_devs, shmemi_device_host_state_t *g_state);
    int (*finalize)(struct shmemi_transport *t,
                        shmemi_device_host_state_t *g_state);

    // data plane, TBD
    void (*rma)(struct shmemi_transport *t, int32_t type, void *dst, void *src, size_t size, int32_t pe);
    void (*amo)(struct shmemi_transport *t, int32_t type, void *dst, void *src, size_t size, int32_t pe);
    void (*quiet)(struct shmemi_transport *t);
    void (*fence)(struct shmemi_transport *t);
    int32_t     logical_dev_id;
    int32_t     dev_id;
} shmemi_transport_t;

typedef struct {
    int32_t pe, npes;

    shmemi_bootstrap_mpi_options_t mpi_options;
    shmemi_bootstrap_uid_options_t uid_options;
    
    // other options
    bool rdma_enabled;
} shmemi_options_t;

// host only state
typedef struct {
    // typedef void *aclrtStream; as in https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_1355.html
    void *default_stream;
    // using TEventID = int8_t; as in https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0181.html
    int8_t default_event_id;
    uint32_t default_block_num;

    // topo
    int32_t *transport_map;                 /* npes * npes, 2D-Array, point-to-connectivity. */
    shmemi_transport_pe_info *pe_info;      /* All pe's host info, need to build transports. */

    shmemi_options_t options;

    shmemi_bootstrap_handle_t *boot_handle;
    shmemi_transport_t choosen_transports[SHMEM_MAX_TRANSPORT_NUM];
    int32_t num_choosen_transport;
} shmemi_host_trans_state_t;

extern shmemi_bootstrap_handle_t g_boot_handle;
extern shmemi_host_trans_state_t g_host_state;
#endif // SHMEMI_HOST_TYPES_H