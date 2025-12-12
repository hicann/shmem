/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <netdb.h>
#include <arpa/inet.h>
#include <functional>

#include "acl/acl.h"
#include "aclshmemi_host_common.h"
#include "host/aclshmem_host_def.h"

using namespace std;

#define DEFAULT_MY_PE (-1)
#define DEFAULT_N_PES (-1)

constexpr int DEFAULT_FLAG = 0;
constexpr int DEFAULT_ID = 0;
constexpr int DEFAULT_TIMEOUT = 120;
constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;

// initializer
#define SHMEM_DEVICE_HOST_STATE_INITIALIZER                                             \
    {                                                                                   \
        (1 << 16) + sizeof(shmemi_device_host_state_t), /* version */                   \
            (DEFAULT_MY_PE),                            /* mype */                      \
            (DEFAULT_N_PES),                            /* npes */                      \
            NULL,                                       /* heap_base */                 \
            NULL,                                       /* host_p2p_heap_base */        \
            NULL,                                       /* host_rdma_heap_base */       \
            NULL,                                       /* host_sdma_heap_base */       \
            NULL,                                       /* p2p_heap_device_base */      \
            NULL,                                       /* sdma_heap_device_base */     \
            NULL,                                       /* roce_heap_device_base */     \
            {},                                         /* topo_list */                 \
            SIZE_MAX,                                   /* heap_size */                 \
            {NULL},                                     /* team_pools */                \
            0,                                          /* sync_pool */                 \
            0,                                          /* sync_counter */              \
            0,                                          /* core_sync_pool */            \
            0,                                          /* core_sync_counter */         \
            false,                                      /* shmem_is_shmem_initialized */\
            false,                                      /* shmem_is_shmem_created */    \
            {0, 16 * 1024, 0},                          /* shmem_mte_config */          \
            0,                                          /* qp_info */                   \
    }

shmemi_device_host_state_t g_state = SHMEM_DEVICE_HOST_STATE_INITIALIZER;
shmemi_host_state_t g_state_host = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM};
shmem_init_attr_t g_attr;
shmemx_uniqueid_t default_flag_uid;

static bool g_attr_init = false;
static char g_ipport[SHMEM_MAX_IP_PORT_LEN] = {0};

int32_t version_compatible()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_options_init()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_state_init_attr(shmem_init_attr_t *attributes)
{
    int32_t status = SHMEM_SUCCESS;
    g_state.mype = attributes->my_rank;
    g_state.npes = attributes->n_ranks;
    g_state.heap_size = attributes->local_mem_size + SHMEM_EXTRA_SIZE;

    aclrtStream stream = nullptr;
    SHMEM_CHECK_RET(aclrtCreateStream(&stream));
    g_state_host.default_stream = stream;
    g_state_host.default_event_id = DEFAULT_TEVENT;
    g_state_host.default_block_num = DEFAULT_BLOCK_NUM;
    return status;
}

int32_t check_attr(shmem_init_attr_t *attributes)
{
    if ((attributes->my_rank < 0) || (attributes->n_ranks <= 0)) {
        SHM_LOG_ERROR("my_rank:" << attributes->my_rank << " and n_ranks: " << attributes->n_ranks
                                 << " cannot be less 0 , n_ranks still cannot be equal 0");
        return SHMEM_INVALID_VALUE;
    } else if (attributes->n_ranks > SHMEM_MAX_RANKS) {
        SHM_LOG_ERROR("n_ranks: " << attributes->n_ranks << " cannot be more than " << SHMEM_MAX_RANKS);
        return SHMEM_INVALID_VALUE;
    } else if (attributes->my_rank >= attributes->n_ranks) {
        SHM_LOG_ERROR("n_ranks:" << attributes->n_ranks << " cannot be less than my_rank:" << attributes->my_rank);
        return SHMEM_INVALID_PARAM;
    } else if (attributes->local_mem_size <= 0) {
        SHM_LOG_ERROR("local_mem_size:" << attributes->local_mem_size << " cannot be less or equal 0");
        return SHMEM_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}

shmemi_init_base* init_manager;

int32_t shmemi_control_barrier_all()
{
    return shmemi_control_barrier_all_default(g_boot_handle);
}

int32_t update_device_state()
{
    return init_manager->update_device_state((void *)&g_state, sizeof(shmemi_device_host_state_t));
}

int32_t shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)
{
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    attributes->option_attr.data_op_engine_type = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)
{
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    attributes->option_attr.shm_init_timeout = value;
    attributes->option_attr.shm_create_timeout = value;
    attributes->option_attr.control_operation_timeout = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes)
{
    SHM_ASSERT_RETURN(local_mem_size <= SHMEM_MAX_LOCAL_SIZE, SHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_ranks <= SHMEM_MAX_RANKS, SHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_rank < SHMEM_MAX_RANKS, SHMEM_INVALID_VALUE);
    *attributes = &g_attr;
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(g_ipport) - 1);

        std::copy_n(ip_port, ip_len, g_ipport);
        g_ipport[ip_len] = '\0';
        std::copy_n(g_ipport, ip_len, g_attr.ip_port);
        if (g_ipport[0] == '\0') {
            SHM_LOG_ERROR("my_rank:" << my_rank << " g_ipport is nullptr!");
            return SHMEM_INVALID_VALUE;
        }
    } else {
        SHM_LOG_WARN("init with my_rank:" << my_rank << " ip_port is nullptr!");
    }

    int attr_version = (1 << 16) + sizeof(shmem_init_attr_t);
    g_attr.my_rank = my_rank;
    g_attr.n_ranks = n_ranks;
    g_attr.ip_port[ip_len] = '\0';
    g_attr.local_mem_size = local_mem_size;
    g_attr.option_attr = {attr_version, SHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    g_attr.comm_args = reinterpret_cast<void *>(&default_flag_uid);
    shmemx_bootstrap_uid_state_t *uid_args = (shmemx_bootstrap_uid_state_t *)(g_attr.comm_args);
    uid_args->rank = my_rank;
    uid_args->nranks = n_ranks;
    g_attr_init = true;
    return SHMEM_SUCCESS;
}

int32_t shmem_init_status(void)
{
    if (!g_state.is_shmem_created)
        return SHMEM_STATUS_NOT_INITIALIZED;
    else if (!g_state.is_shmem_initialized)
        return SHMEM_STATUS_SHM_CREATED;
    else if (g_state.is_shmem_initialized)
        return SHMEM_STATUS_IS_INITIALIZED;
    else
        return SHMEM_STATUS_INVALID;
}

int shmemx_set_attr_uniqueid_args(const int my_rank, const int n_ranks, const int64_t local_mem_size,
                                    const shmemx_uniqueid_t *uid,
                                    shmem_init_attr_t **shmem_attr) {
    /* Save to uid_args */
    *shmem_attr = &g_attr;
    shmemx_bootstrap_uid_state_t *uid_args = (shmemx_bootstrap_uid_state_t *)(uid);
    uid_args->rank = my_rank;
    uid_args->nranks = n_ranks;
    void * comm_args = reinterpret_cast<void *>(uid_args);
    g_attr.comm_args = comm_args;
    g_attr.my_rank = my_rank;
    g_attr.n_ranks = n_ranks;
    g_attr.local_mem_size = local_mem_size;
    g_attr_init = true;
    return SHMEM_SUCCESS;
}

int32_t shmem_init_attr(shmemx_bootstrap_t bootstrap_flags, shmem_init_attr_t *attributes)
{
    int32_t ret;
    SHMEM_CHECK_RET(shmem_set_log_level(shm::ERROR_LEVEL));

    // config init
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    SHMEM_CHECK_RET(check_attr(attributes), "An error occurred while checking the initialization attributes. Please check the initialization parameters.");
    SHMEM_CHECK_RET(version_compatible(), "SHMEM Version mismatch.");
    SHMEM_CHECK_RET(shmemi_options_init());

    // shmem basic init
    SHM_LOG_INFO("The current backend is SHMEM default.");
    // bootstrap init
    SHMEM_CHECK_RET(shmemi_bootstrap_init(bootstrap_flags, attributes));
    init_manager = new shmemi_init_default(attributes, &g_state);

    SHMEM_CHECK_RET(shmemi_state_init_attr(attributes));
    SHMEM_CHECK_RET(init_manager->init_device_state());
    SHMEM_CHECK_RET(init_manager->reserve_heap());
    SHMEM_CHECK_RET(init_manager->transport_init());
    SHMEM_CHECK_RET(init_manager->setup_heap());
    
    // shmem submodules init
    SHMEM_CHECK_RET(memory_manager_initialize(g_state.heap_base, g_state.heap_size));
    SHMEM_CHECK_RET(shmemi_team_init(g_state.mype, g_state.npes));
    SHMEM_CHECK_RET(shmemi_sync_init());
    g_state.is_shmem_initialized = true;
    SHMEM_CHECK_RET(update_device_state());
    SHMEM_CHECK_RET(shmemi_control_barrier_all());
    SHM_LOG_INFO("SHMEM init success.");
    return SHMEM_SUCCESS;
}

int32_t shmem_finalize()
{
    SHM_LOG_INFO("The pe: " << shmem_my_pe() << " begins to finalize.");
    // shmem submodules finalize
    SHMEM_CHECK_RET(shmemi_team_finalize());

    // shmem basic finalize
    SHMEM_CHECK_RET(init_manager->remove_heap());
    SHMEM_CHECK_RET(init_manager->transport_finalize());
    SHMEM_CHECK_RET(init_manager->release_heap());
    SHMEM_CHECK_RET(init_manager->finalize_device_state());
    delete init_manager;
    shmemi_bootstrap_finalize();
    SHM_LOG_INFO("The pe: " << shmem_my_pe() << " finalize success.");
    return SHMEM_SUCCESS;
}

void shmem_info_get_version(int *major, int *minor)
{
    SHM_ASSERT_RET_VOID(major != nullptr && minor != nullptr);
    *major = SHMEM_MAJOR_VERSION;
    *minor = SHMEM_MINOR_VERSION;
}

void shmem_info_get_name(char *name)
{
    SHM_ASSERT_RET_VOID(name != nullptr);
    std::ostringstream oss;
    oss << "SHMEM v" << SHMEM_VENDOR_MAJOR_VER << "." << SHMEM_VENDOR_MINOR_VER << "." << SHMEM_VENDOR_PATCH_VER;
    auto version_str = oss.str();
    size_t i;
    for (i = 0; i < SHMEM_MAX_NAME_LEN - 1 && version_str[i] != '\0'; i++) {
        name[i] = version_str[i];
    }
    name[i] = '\0';
}

int32_t shmem_get_uniqueid_default(shmemx_uniqueid_t *uid)
{
    int status = 0;
    SHMEM_CHECK_RET(shmemi_options_init(), "Bootstrap failed during the preloading step.");
    SHMEM_CHECK_RET(shmemi_bootstrap_pre_init(SHMEMX_INIT_WITH_UNIQUEID, &g_boot_handle), "Get uniqueid failed during the bootstrap preloading step.");

    if (g_boot_handle.pre_init_ops) {
        SHMEM_CHECK_RET(g_boot_handle.pre_init_ops->get_unique_id((void *)uid), "Get uniqueid failed during the get uniqueid step.");
    } else {
        SHM_LOG_ERROR("Pre_init_ops is empty, unique_id cannot be obtained.");
        status = SHMEM_INVALID_PARAM;
    }

    return (status);
}

int32_t shmem_get_uniqueid(shmemx_uniqueid_t *uid){
    shmem_set_log_level(shm::ERROR_LEVEL);
    *uid = SHMEM_UNIQUEID_INITIALIZER;
    SHMEM_CHECK_RET(shmem_get_uniqueid_default(uid), "shmem_get_uniqueid failed, backend: default");
    return SHMEM_SUCCESS;

}

int32_t shmemi_get_uniqueid_static_magic(shmemx_uniqueid_t *uid, bool is_root) {
    shmem_set_log_level(shm::ERROR_LEVEL);
    *uid = SHMEM_UNIQUEID_INITIALIZER;
    int status = 0;
    SHMEM_CHECK_RET(shmemi_options_init(), "Bootstrap failed during the preloading step.");
    SHMEM_CHECK_RET(shmemi_bootstrap_pre_init(SHMEMX_INIT_WITH_UNIQUEID, &g_boot_handle), "Get uniqueid failed during the bootstrap preloading step.");

    if (g_boot_handle.pre_init_ops) {
        SHMEM_CHECK_RET(g_boot_handle.pre_init_ops->get_unique_id_static_magic((void *)uid, is_root), "Get uniqueid failed during the get uniqueid step.");
    } else {
        SHM_LOG_ERROR("Pre_init_ops is empty, unique_id cannot be obtained.");
        status = SHMEM_INVALID_PARAM;
    }
    return SHMEM_SUCCESS;
}


int32_t shmem_set_log_level(int level)
{
    // use env first, input level secondly, user may change level from env instead call func
    const char *in_level = std::getenv("SHMEM_LOG_LEVEL");
    if (in_level != nullptr) {
        auto tmp_level = std::string(in_level);
        if (tmp_level == "DEBUG") {
            level = shm::DEBUG_LEVEL;
        } else if (tmp_level == "INFO") {
            level = shm::INFO_LEVEL;
        } else if (tmp_level == "WARN") {
            level = shm::WARN_LEVEL;
        } else if (tmp_level == "ERROR") {
            level = shm::ERROR_LEVEL;
        } else if (tmp_level == "FATAL") {
            level = shm::FATAL_LEVEL;
        }
    }
    
    return shm::shm_out_logger::Instance().set_log_level(static_cast<shm::log_level>(level));
}

int32_t shmem_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
    return SHMEM_SUCCESS;
}

void shmem_rank_exit(int status)
{
    SHM_LOG_DEBUG("shmem_rank_exit is work ,status: " << status);
    exit(status);
}

int32_t shmem_set_config_store_tls_key(const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const shmem_decrypt_handler decrypt_handler)
{
    return SHMEM_SUCCESS;
}

int32_t shmem_set_extern_logger(void (*func)(int level, const char *msg))
{
    return SHMEM_SUCCESS;
}

void shmem_global_exit(int status)
{
    return;
}