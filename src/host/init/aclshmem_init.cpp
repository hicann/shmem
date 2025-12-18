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

#ifdef BACKEND_HYBM
#include "store_factory.h"
#endif

using namespace std;

#define DEFAULT_MY_PE (-1)
#define DEFAULT_N_PES (-1)

constexpr int DEFAULT_FLAG = 0;
constexpr int DEFAULT_ID = 0;
constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;

// initializer
#define ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER                                             \
    {                                                                                   \
        (1 << 16) + sizeof(aclshmem_device_host_state_t), /* version */                   \
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
            false,                                      /* aclshmem_is_aclshmem_initialized */\
            false,                                      /* aclshmem_is_aclshmem_created */    \
            {0, 16 * 1024, 0},                          /* aclshmem_mte_config */          \
            0,                                          /* qp_info */                   \
    }

aclshmem_device_host_state_t g_state = ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER;
aclshmem_host_state_t g_state_host = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM};

int32_t version_compatible()
{
    int32_t status = ACLSHMEM_SUCCESS;
    return status;
}

int32_t aclshmemi_options_init()
{
    int32_t status = ACLSHMEM_SUCCESS;
    return status;
}

int32_t aclshmemi_state_init_attr(aclshmemx_init_attr_t *attributes)
{
    int32_t status = ACLSHMEM_SUCCESS;
    g_state.mype = attributes->my_pe;
    g_state.npes = attributes->n_pes;
    g_state.heap_size = attributes->local_mem_size + ACLSHMEM_EXTRA_SIZE;

    aclrtStream stream = nullptr;
    ACLSHMEM_CHECK_RET(aclrtCreateStream(&stream));
    g_state_host.default_stream = stream;
    g_state_host.default_event_id = DEFAULT_TEVENT;
    g_state_host.default_block_num = DEFAULT_BLOCK_NUM;
    return status;
}

int32_t check_attr(aclshmemx_init_attr_t *attributes)
{
    if ((attributes->my_pe < 0) || (attributes->n_pes <= 0)) {
        SHM_LOG_ERROR("my_pe:" << attributes->my_pe << " and n_pes: " << attributes->n_pes
                                 << " cannot be less 0 , n_pes still cannot be equal 0");
        return ACLSHMEM_INVALID_VALUE;
    } else if (attributes->n_pes > ACLSHMEM_MAX_PES) {
        SHM_LOG_ERROR("n_pes: " << attributes->n_pes << " cannot be more than " << ACLSHMEM_MAX_PES);
        return ACLSHMEM_INVALID_VALUE;
    } else if (attributes->my_pe >= attributes->n_pes) {
        SHM_LOG_ERROR("n_pes:" << attributes->n_pes << " cannot be less than my_pe:" << attributes->my_pe);
        return ACLSHMEM_INVALID_PARAM;
    } else if (attributes->local_mem_size <= 0) {
        SHM_LOG_ERROR("local_mem_size:" << attributes->local_mem_size << " cannot be less or equal 0");
        return ACLSHMEM_INVALID_VALUE;
    }
    return ACLSHMEM_SUCCESS;
}

aclshmemi_init_base* init_manager;

int32_t aclshmemi_control_barrier_all()
{
#ifdef BACKEND_HYBM
    return init_manager->aclshmemi_control_barrier_all();
#else
    return aclshmemi_control_barrier_all_default(g_boot_handle);
#endif
}

int32_t update_device_state()
{
    return init_manager->update_device_state((void *)&g_state, sizeof(aclshmem_device_host_state_t));
}


int32_t aclshmemx_init_status(void)
{
    if (!g_state.is_aclshmem_created)
        return ACLSHMEM_STATUS_NOT_INITIALIZED;
    else if (!g_state.is_aclshmem_initialized)
        return ACLSHMEM_STATUS_SHM_CREATED;
    else if (g_state.is_aclshmem_initialized)
        return ACLSHMEM_STATUS_IS_INITIALIZED;
    else
        return ACLSHMEM_STATUS_INVALID;
}

int aclshmemx_set_attr_uniqueid_args(int my_pe, int n_pes, int64_t local_mem_size,
                                    aclshmemx_uniqueid_t *uid,
                                    aclshmemx_init_attr_t *aclshmem_attr) {
    /* Save to uid_args */
    SHM_ASSERT_RETURN(local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_pes <= ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_pe < ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    aclshmemi_bootstrap_uid_state_t *uid_args = (aclshmemi_bootstrap_uid_state_t *)(uid);
    uid_args->my_pe = my_pe;
    uid_args->n_pes = n_pes;
    void * comm_args = reinterpret_cast<void *>(uid_args);
    aclshmem_attr->comm_args = comm_args;
    aclshmem_attr->my_pe = my_pe;
    aclshmem_attr->n_pes = n_pes;
    aclshmem_attr->local_mem_size = local_mem_size;
#ifdef BACKEND_HYBM
    std::string ipPort;
    if (uid_args->addr.type == ADDR_IPv6) {
        char ipStr[INET6_ADDRSTRLEN] = {0};
        if (inet_ntop(AF_INET6, &(uid_args->addr.addr.addr6.sin6_addr), ipStr, sizeof(ipStr)) == nullptr) {
            SHM_LOG_ERROR("inet_ntop failed for IPv6");
            return ACLSHMEM_INNER_ERROR;
        }
        uint16_t port = ntohs(uid_args->addr.addr.addr6.sin6_port);
        ipPort = "tcp6://[" + std::string(ipStr) + "]:" + std::to_string(port);
    } else {
        char ipStr[INET_ADDRSTRLEN] = {0};
        if (inet_ntop(AF_INET, &(uid_args->addr.addr.addr4.sin_addr), ipStr, sizeof(ipStr)) == nullptr) {
            SHM_LOG_ERROR("inet_ntop failed for IPv4");
            return ACLSHMEM_INNER_ERROR;
        }
        uint16_t port = ntohs(uid_args->addr.addr.addr4.sin_port);
        ipPort = "tcp://" + std::string(ipStr) + ":" + std::to_string(port);
    }
    std::copy(ipPort.begin(), ipPort.end(), aclshmem_attr->ip_port);
    aclshmem_attr->ip_port[ipPort.size()] = '\0';
    int attr_version = static_cast<int>((1 << 16) + sizeof(aclshmemx_init_attr_t));
    aclshmem_attr->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, 0};
     aclshmem_attr->option_attr.sockFd = uid_args->inner_sockFd;
    SHM_LOG_INFO("extract ip port:" << ipPort);
#endif
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemx_init_attr(aclshmemx_bootstrap_t bootstrap_flags, aclshmemx_init_attr_t *attributes)
{
    int32_t ret;
    // config init
    SHM_ASSERT_RETURN(attributes != nullptr, ACLSHMEM_INVALID_PARAM);
    ACLSHMEM_CHECK_RET(aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL));
    ACLSHMEM_CHECK_RET(check_attr(attributes), "An error occurred while checking the initialization attributes. Please check the initialization parameters.");
    ACLSHMEM_CHECK_RET(version_compatible(), "ACLSHMEM Version mismatch.");
    ACLSHMEM_CHECK_RET(aclshmemi_options_init());
#if defined(BACKEND_HYBM)
    SHM_LOG_INFO("The current backend is HYBM.");
    ACLSHMEM_CHECK_RET(bootstrap_flags != ACLSHMEMX_INIT_WITH_DEFAULT, "The current backend is MF, and the value of bootstrap_flags only supports SHMEMX_INIT_WITH_DEFAULT.", ACLSHMEM_INVALID_PARAM);
    init_manager = new aclshmemi_init_hybm(attributes, attributes->ip_port, &g_state);
#else
    // aclshmem basic init
    SHM_LOG_INFO("The current backend is ACLSHMEM default.");
    // bootstrap init
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_init(bootstrap_flags, attributes));
    init_manager = new aclshmemi_init_default(attributes, &g_state);
#endif
    ACLSHMEM_CHECK_RET(aclshmemi_state_init_attr(attributes));
    ACLSHMEM_CHECK_RET(init_manager->init_device_state());
    ACLSHMEM_CHECK_RET(init_manager->reserve_heap());
    ACLSHMEM_CHECK_RET(init_manager->transport_init());
    ACLSHMEM_CHECK_RET(init_manager->setup_heap());
    
    // aclshmem submodules init
    ACLSHMEM_CHECK_RET(memory_manager_initialize(g_state.heap_base, g_state.heap_size));
    ACLSHMEM_CHECK_RET(aclshmemi_team_init(g_state.mype, g_state.npes));
    ACLSHMEM_CHECK_RET(aclshmemi_sync_init());
    g_state.is_aclshmem_initialized = true;
    ACLSHMEM_CHECK_RET(update_device_state());
    ACLSHMEM_CHECK_RET(aclshmemi_control_barrier_all());
    SHM_LOG_INFO("ACLSHMEM init success.");
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmem_finalize()
{
    SHM_LOG_INFO("The pe: " << aclshmem_my_pe() << " begins to finalize.");
    // aclshmem submodules finalize
    ACLSHMEM_CHECK_RET(aclshmemi_team_finalize());

    // aclshmem basic finalize
    ACLSHMEM_CHECK_RET(init_manager->remove_heap());
    ACLSHMEM_CHECK_RET(init_manager->transport_finalize());
    ACLSHMEM_CHECK_RET(init_manager->release_heap());
    ACLSHMEM_CHECK_RET(init_manager->finalize_device_state());
    delete init_manager;
#ifdef BACKEND_HYBM
#else
    aclshmemi_bootstrap_finalize();
#endif
    SHM_LOG_INFO("The pe: " << aclshmem_my_pe() << " finalize success.");
    return ACLSHMEM_SUCCESS;
}

void aclshmem_info_get_version(int *major, int *minor)
{
    SHM_ASSERT_RET_VOID(major != nullptr && minor != nullptr);
    *major = ACLSHMEM_MAJOR_VERSION;
    *minor = ACLSHMEM_MINOR_VERSION;
}

void aclshmem_info_get_name(char *name)
{
    SHM_ASSERT_RET_VOID(name != nullptr);
    std::ostringstream oss;
    oss << "ACLSHMEM v" << ACLSHMEM_VENDOR_MAJOR_VER << "." << ACLSHMEM_VENDOR_MINOR_VER << "." << ACLSHMEM_VENDOR_PATCH_VER;
    auto version_str = oss.str();
    size_t i;
    for (i = 0; i < ACLSHMEM_MAX_NAME_LEN - 1 && version_str[i] != '\0'; i++) {
        name[i] = version_str[i];
    }
    name[i] = '\0';
}

int32_t aclshmemi_get_uniqueid_default(aclshmemx_uniqueid_t *uid)
{
    int status = 0;
    ACLSHMEM_CHECK_RET(aclshmemi_options_init(), "Bootstrap failed during the preloading step.");
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_pre_init(ACLSHMEMX_INIT_WITH_UNIQUEID, &g_boot_handle), "Get uniqueid failed during the bootstrap preloading step.");

    if (g_boot_handle.pre_init_ops) {
        ACLSHMEM_CHECK_RET(g_boot_handle.pre_init_ops->get_unique_id((void *)uid), "Get uniqueid failed during the get uniqueid step.");
    } else {
        SHM_LOG_ERROR("Pre_init_ops is empty, unique_id cannot be obtained.");
        status = ACLSHMEM_INVALID_PARAM;
    }

    return (status);
}

int32_t aclshmemx_get_uniqueid(aclshmemx_uniqueid_t *uid){
    aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL);
    *uid = ACLSHMEM_UNIQUEID_INITIALIZER;
#ifdef BACKEND_HYBM
    ACLSHMEM_CHECK_RET(aclshmemi_get_uniqueid_hybm(uid), "shmem_get_uniqueid failed, backend: hybm");
#else
    ACLSHMEM_CHECK_RET(aclshmemi_get_uniqueid_default(uid), "aclshmemx_get_uniqueid failed, backend: default");
#endif
    return ACLSHMEM_SUCCESS;

}

int32_t aclshmemi_get_uniqueid_static_magic(aclshmemx_uniqueid_t *uid, bool is_root) {
    aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL);
    *uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    int status = 0;
    ACLSHMEM_CHECK_RET(aclshmemi_options_init(), "Bootstrap failed during the preloading step.");
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_pre_init(ACLSHMEMX_INIT_WITH_UNIQUEID, &g_boot_handle), "Get uniqueid failed during the bootstrap preloading step.");

    if (g_boot_handle.pre_init_ops) {
        ACLSHMEM_CHECK_RET(g_boot_handle.pre_init_ops->get_unique_id_static_magic((void *)uid, is_root), "Get uniqueid failed during the get uniqueid step.");
    } else {
        SHM_LOG_ERROR("Pre_init_ops is empty, unique_id cannot be obtained.");
        status = ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}


int32_t aclshmemx_set_log_level(int level)
{
    // use env first, input level secondly, user may change level from env instead call func
    const char *in_level = std::getenv("ACLSHMEM_LOG_LEVEL");
    if (in_level != nullptr) {
        auto tmp_level = std::string(in_level);
        if (tmp_level == "DEBUG") {
            level = aclshmem_log::DEBUG_LEVEL;
        } else if (tmp_level == "INFO") {
            level = aclshmem_log::INFO_LEVEL;
        } else if (tmp_level == "WARN") {
            level = aclshmem_log::WARN_LEVEL;
        } else if (tmp_level == "ERROR") {
            level = aclshmem_log::ERROR_LEVEL;
        } else if (tmp_level == "FATAL") {
            level = aclshmem_log::FATAL_LEVEL;
        }
    }
    
    return aclshmem_log::aclshmem_out_logger::Instance().set_log_level(static_cast<aclshmem_log::log_level>(level));
}

int32_t aclshmemx_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
#ifdef BACKEND_HYBM
    return ock::smem::StoreFactory::SetTlsInfo(enable, tls_info, tls_info_len);
#else
    return ACLSHMEM_SUCCESS;
#endif
}

void aclshmem_rank_exit(int status)
{
    SHM_LOG_DEBUG("aclshmem_rank_exit is work ,status: " << status);
    exit(status);
}

int32_t aclshmemx_set_config_store_tls_key(const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const aclshmem_decrypt_handler decrypt_handler)
{
#ifdef BACKEND_HYBM
    return ock::smem::StoreFactory::SetTlsPkInfo(tls_pk, tls_pk_len, tls_pk_pw, tls_pk_pw_len, decrypt_handler);
#else
    return ACLSHMEM_SUCCESS;
#endif
}

int32_t aclshmemx_set_extern_logger(void (*func)(int level, const char *msg))
{
    return ACLSHMEM_SUCCESS;
}

void aclshmem_global_exit(int status)
{
#ifdef BACKEND_HYBM
    init_manager->aclshmemi_global_exit(status);
#else
#endif
}