/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <random>
#include "aclshmemi_init_hybm.h"
#include "aclshmemi_host_common.h"
#ifdef BACKEND_HYBM
#include <fstream>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "host/aclshmem_host_def.h"
#include "utils/aclshmemi_logger.h"
#include "store_factory.h"
#include "hybm_big_mem.h"
#include "sotre_net.h"
#include "store_net_common.h"

constexpr int MIN_PORT = 1024;
constexpr int MAX_PORT = 65536;
constexpr int MAX_ATTEMPTS = 1000;
constexpr int MAX_IFCONFIG_LENGTH = 23;
constexpr int MAX_IP = 48;
constexpr int DEFAULT_IFNAME_LNEGTH = 4;

constexpr int DEFAULT_FLAG = 0;
constexpr int DEFAULT_ID = 0;
constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;

static char g_ipport[ACLSHMEM_MAX_IP_PORT_LEN] = {0};

aclshmemi_init_hybm::aclshmemi_init_hybm(aclshmemx_init_attr_t *attr, char *ipport, aclshmem_device_host_state_t *global_state)
{
    attributes = attr;
    npes = attr->n_pes;
    g_state = global_state;

    auto status = aclrtGetDevice(&device_id);
    if (status != 0) {
        SHM_LOG_ERROR("Get Device_id error");
    }
    strncpy(g_ipport, ipport, ACLSHMEM_MAX_IP_PORT_LEN - 1);
    g_ipport[ACLSHMEM_MAX_IP_PORT_LEN - 1] = '\0';
    g_state = global_state;

    // TODO set tls
    shm::store::StoreFactory::SetTlsInfo(false, nullptr, 0);
}

aclshmemi_init_hybm::~aclshmemi_init_hybm()
{}

int aclshmemi_init_hybm::init_device_state()
{
    int32_t status = ACLSHMEM_SUCCESS;
    int32_t sock_fd = attributes->option_attr.sockFd;

    shm::store::UrlExtraction option;
    std::string url(attributes->ip_port);
    SHM_ASSERT_RETURN(option.ExtractIpPortFromUrl(url) == ACLSHMEM_SUCCESS, ACLSHMEM_INVALID_PARAM);

    if (attributes->my_pe == 0) {
        store_ = shm::store::StoreFactory::CreateStore(option.ip, option.port, true, 0, -1, sock_fd);
        ip_ = option.ip;
        port_ = option.port;
    }
    else {
        store_ = shm::store::StoreFactory::CreateStore(option.ip, option.port, false, attributes->my_pe, 120);
        ip_ = option.ip;
        port_ = option.port;
    }
    // 初始化hybm，分配meta存储空间
    auto ret = hybm_init(device_id, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("init hybm failed, result: " << ret);
        return ACLSHMEM_SMEM_ERROR;
    }
    // 创建groupengine
    std::string prefix = "SHM_(" + std::to_string(DEFAULT_ID) + ")_";
    shm::store::StorePtr store_ptr = shm::store::StoreFactory::PrefixStore(store_, prefix);
    shm::store::SmemGroupOption opt = {(uint32_t)attributes->n_pes, (uint32_t)attributes->my_pe,  120 * 1000U,
                           false, nullptr, nullptr};
    shm::store::SmemGroupEnginePtr group = shm::store::SmemNetGroupEngine::Create(store_ptr, opt);
    SHM_ASSERT_RETURN(group != nullptr, ACLSHMEM_SMEM_ERROR);

    globalGroup_ = group;
    ret = aclshmemi_control_barrier_all();  // 保证所有rank都初始化了
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_control_barrier_allfailed, result: " << ret);
        return ACLSHMEM_SMEM_ERROR;
    }
    // 创建entity
    hybm_options options;
    options.bmType = HYBM_TYPE_AI_CORE_INITIATE;
    options.memType = HYBM_MEM_TYPE_DEVICE;
    options.bmDataOpType = static_cast<hybm_data_op_type>(HYBM_DOP_TYPE_MTE);
    if (attributes->option_attr.data_op_engine_type & ACLSHMEM_DATA_OP_ROCE) {
        auto temp = static_cast<uint32_t>(options.bmDataOpType) | HYBM_DOP_TYPE_DEVICE_RDMA;
        options.bmDataOpType = static_cast<hybm_data_op_type>(temp);
    }
    options.bmScope = HYBM_SCOPE_CROSS_NODE;
    options.rankCount = attributes->n_pes;
    options.rankId = attributes->my_pe;
    options.singleRankVASpace = g_state->heap_size;
    options.preferredGVA = 0;
    options.role = HYBM_ROLE_PEER;
    options.globalUniqueAddress = true;
    std::string defaultNic = "10002";
    std::copy_n(defaultNic.c_str(), defaultNic.size() + 1, options.nic);

    auto entity = hybm_create_entity(DEFAULT_ID << 1, &options, 0);
    if (entity == nullptr) {
        SHM_LOG_ERROR("create entity failed");
        return ACLSHMEM_SMEM_ERROR;
    }
    entity_ = entity;
    inited_ = true;
    SHM_LOG_INFO("init_device_state success. world_size: " << attributes->n_pes);
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::update_device_state(void* host_ptr, size_t size)
{
    int32_t ptr_size = g_state->npes * sizeof(void *);
    ACLSHMEM_CHECK_RET(aclrtMemcpy(g_state->device_p2p_heap_base, ptr_size, g_state->host_p2p_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(g_state->device_rdma_heap_base, ptr_size, g_state->host_rdma_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLSHMEM_CHECK_RET(aclrtMemcpy(g_state->device_sdma_heap_base, ptr_size, g_state->host_sdma_heap_base, ptr_size, ACL_MEMCPY_HOST_TO_DEVICE));

    auto ret = hybm_set_extra_context(entity_, host_ptr, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("hybm_set_extra_context failed, ret: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::finalize_device_state()
{
    if (!inited_) {
        SHM_LOG_WARN("smem shm not initialized yet");
        return ACLSHMEM_INNER_ERROR;
    }
    hybm_destroy_entity(entity_, 0);
    hybm_uninit();
    inited_ = false;
    store_ = nullptr;

    SHM_LOG_INFO("shmemi uninit finished");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::reserve_heap()
{
    void *gva = nullptr;

    auto ret = hybm_reserve_mem_space(entity_, 0, &gva);
    if (ret != 0 || gva == nullptr) {
        SHM_LOG_ERROR("reserve mem failed, result: " << ret);
        return ACLSHMEM_SMEM_ERROR;
    }
    gva_ = gva;
    SHM_LOG_INFO("reserve_heap success.");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::reach_info_init(void *&gva)
{
    for (int32_t i = 0; i < g_state->npes; i++) {
        hybm_data_op_type reaches_types;
        auto ret = hybm_entity_reach_types(entity_, i, reaches_types, 0);
        if (ret != 0) {
            SHM_LOG_ERROR("hybm_entity_reach_types failed: " << ret);
            return ACLSHMEM_SMEM_ERROR;
        }

        g_state->host_p2p_heap_base[i] = (void *)((uintptr_t)gva + g_state->heap_size * static_cast<uint32_t>(i));
        if (reaches_types & HYBM_DOP_TYPE_MTE) {
            g_state->topo_list[i] |= ACLSHMEM_TRANSPORT_MTE;
        }
        if (reaches_types & HYBM_DOP_TYPE_DEVICE_RDMA) {
            g_state->topo_list[i] |= ACLSHMEM_TRANSPORT_ROCE;
        }
        g_state->host_sdma_heap_base[i] = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::exchange_slice()
{
    // export memory
    hybm_exchange_info ex_info;
    bzero(&ex_info, sizeof(ex_info));
    auto ret = hybm_export(entity_, slice_, 0, &ex_info);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm export slice failed, result: " << ret);
        return ret;
    }

    hybm_exchange_info all_ex_info[g_state->npes];
    ret = globalGroup_->GroupAllGather((char *)&ex_info, sizeof(hybm_exchange_info), (char *)all_ex_info,
                                       sizeof(hybm_exchange_info) * g_state->npes);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm gather export slice failed, result: " << ret);
        return ret;
    }
    // import memory
    ret = hybm_import(entity_, all_ex_info, g_state->npes, nullptr, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm import failed, result: " << ret);
        return ret;
    }

    ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("hybm barrier for slice failed, result: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::exchange_entity()
{
    // export entity
    hybm_exchange_info ex_info;
    bzero(&ex_info, sizeof(ex_info));
    auto ret = hybm_export(entity_, nullptr, 0, &ex_info);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm export entity failed, result: " << ret);
        return ret;
    }

    if (ex_info.descLen == 0) {
        return ACLSHMEM_SUCCESS;
    }

    hybm_exchange_info all_ex_info[g_state->npes];
    ret = globalGroup_->GroupAllGather((char *)&ex_info, sizeof(hybm_exchange_info), (char *)all_ex_info,
                                       sizeof(hybm_exchange_info) * g_state->npes);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm gather export entity failed, result: " << ret);
        return ret;
    }
    // import entity
    ret = hybm_import(entity_, all_ex_info, g_state->npes, nullptr, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm import entity failed, result: " << ret);
        return ret;
    }

    ret = aclshmemi_control_barrier_all();
    if (ret != 0) {
        SHM_LOG_ERROR("hybm barrier for slice failed, result: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::setup_heap()
{
    // alloc memory
    auto slice = hybm_alloc_local_memory(entity_, HYBM_MEM_TYPE_DEVICE, g_state->heap_size, 0);
    if (slice == nullptr) {
        SHM_LOG_ERROR("alloc local mem failed, size: " << g_state->heap_size);
        return ACLSHMEM_SMEM_ERROR;
    }
    slice_ = slice;
    auto ret = exchange_slice();
    if (ret != 0) {
        SHM_LOG_ERROR("exchange slice failed, result: " << ret);
        return ret;
    }

    ret = exchange_entity();
    if (ret != 0) {
        SHM_LOG_ERROR("exchange slice failed, result: " << ret);
        return ret;
    }

    // mmap memory
    ret = hybm_mmap(entity_, 0);
    if (ret != 0) {
        SHM_LOG_ERROR("hybm mmap failed, result: " << ret);
        return ret;
    }
    g_state->heap_base = (void *)((uintptr_t)gva_ + g_state->heap_size * static_cast<uint32_t>(attributes->my_pe));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&g_state->host_p2p_heap_base, g_state->npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&g_state->host_rdma_heap_base, g_state->npes * sizeof(void *)));
    ACLSHMEM_CHECK_RET(aclrtMallocHost((void **)&g_state->host_sdma_heap_base, g_state->npes * sizeof(void *)));

    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&g_state->device_p2p_heap_base, g_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&g_state->device_rdma_heap_base, g_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ACLSHMEM_CHECK_RET(aclrtMalloc((void **)&g_state->device_sdma_heap_base, g_state->npes * sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    ret = reach_info_init(gva_);
    if (ret != 0) {
        SHM_LOG_ERROR("reach_info_init failed, result: " << ret);
        return ret;
    }
    if (g_ipport[0] != '\0') {
        g_ipport[0] = '\0';
        bzero(attributes->ip_port, sizeof(attributes->ip_port));
    } else {
        SHM_LOG_WARN("my_rank:" << attributes->my_pe << " g_ipport is released in advance!");
        bzero(attributes->ip_port, sizeof(attributes->ip_port));
    }
    g_state->is_aclshmem_created = true;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::remove_heap()
{
    hybm_uninit();    
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::release_heap()
{
    if (g_state->host_p2p_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(g_state->host_p2p_heap_base));
    }
    if (g_state->host_p2p_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(g_state->host_rdma_heap_base));
    }
    if (g_state->host_p2p_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFreeHost(g_state->host_sdma_heap_base));
    }
    if (g_state->device_p2p_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFree(g_state->device_p2p_heap_base));
    }
    if (g_state->device_rdma_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFree(g_state->device_rdma_heap_base));
    }
    if (g_state->device_sdma_heap_base != nullptr) {
        ACLSHMEM_CHECK_RET(aclrtFree(g_state->device_sdma_heap_base));
    }

    if (globalGroup_ != nullptr) {
        globalGroup_ = nullptr;
    }

    uint32_t flags = 0;
    if (entity_ != nullptr && slice_ != nullptr) {
        hybm_free_local_memory(entity_, slice_, 1, flags);
    }
    auto reserved_mem = gva_;
    auto ret = hybm_unreserve_mem_space(entity_, 0, reserved_mem);
    if (ret != 0) {
        SHM_LOG_WARN("unreserve mem space failed: " << ret);
    }
    gva_ = nullptr;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::transport_init()
{
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_init_hybm::transport_finalize()
{
    return ACLSHMEM_SUCCESS;
}

void aclshmemi_init_hybm::aclshmemi_global_exit(int status)
{
    if (globalGroup_ == nullptr) {
        SHM_LOG_ERROR("Group is NULL");
        return;
    }
    globalGroup_->GroupBroadcastExit(status);
}

int aclshmemi_init_hybm::aclshmemi_control_barrier_all()
{
    if (globalGroup_ == nullptr) {
        SHM_LOG_ERROR("Group is NULL");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = globalGroup_->GroupBarrier();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Group barrier timeout or store failure");
        return ACLSHMEM_SMEM_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t shmem_get_uid_magic(shmemx_bootstrap_uid_state_t *innerUId)
{
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) {
        SHM_LOG_ERROR("open random failed");
        return ACLSHMEM_INNER_ERROR;
    }

    urandom.read(reinterpret_cast<char *>(&innerUId->magic), sizeof(innerUId->magic));
    if (urandom.fail()) {
        SHM_LOG_ERROR("read random failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("init magic id to " << innerUId->magic);
    return ACLSHMEM_SUCCESS;
}

int32_t bind_tcp_port_v4(int &sockfd, int port, shmemx_bootstrap_uid_state_t *innerUId, char *ip_str)
{
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v4 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v4, sizeof(on_v4)) == 0) {
            innerUId->addr.addr.addr4.sin_port = htons(port);
            sockaddr *cur_addr = reinterpret_cast<sockaddr *>(&innerUId->addr.addr.addr4);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr4)) == 0) {
                SHM_LOG_INFO("bind ipv4 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket opt fail:" << errno << ","  << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket fail:" << errno << ", " << ip_str << ":" << port);
    }
    return -1;
}

int32_t bind_tcp_port_v6(int &sockfd, int port, shmemx_bootstrap_uid_state_t *innerUId, char *ip_str)
{
    sockfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v6 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v6, sizeof(on_v6)) == 0) {
            innerUId->addr.addr.addr6.sin6_port = htons(port);
            sockaddr *cur_addr = reinterpret_cast<sockaddr *>(&innerUId->addr.addr.addr6);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr6)) == 0) {
                SHM_LOG_INFO("bind ipv6 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket6 fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket6 opt fail:" << errno << "," << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket6 fail:" << errno << "," << ip_str << ":" << port);
    }
    return -1;
}

int32_t aclshmemi_get_port_magic(shmemx_bootstrap_uid_state_t *innerUId, char *ip_str)
{
    static std::random_device rd;
    const int min_port = MIN_PORT;
    const int max_port = MAX_PORT;
    const int max_attempts = MAX_ATTEMPTS;
    const int offset_bit = 32;
    uint64_t seed = 1;
    seed |= static_cast<uint64_t>(getpid()) << offset_bit;
    seed |= static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFF);
    static std::mt19937_64 gen(seed);
    std::uniform_int_distribution<> dis(min_port, max_port);

    int sockfd = -1;
    int32_t ret;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = dis(gen);
        if (innerUId->addr.type == ADDR_IPv4) {
            ret = bind_tcp_port_v4(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        } else {
            ret = bind_tcp_port_v6(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        }
    }
    SHM_LOG_ERROR("Not find a available tcp port");
    return -1;
}

int32_t aclshmemi_using_env_port(aclshmemi_bootstrap_uid_state_t *innerUId, char *ip_str, uint16_t envPort)
{
    if (envPort < MIN_PORT) {   // envPort > MAX_PORT always false
        SHM_LOG_ERROR("env port is invalid. " << envPort);
        return ACLSHMEM_INVALID_PARAM;
    }

    int sockfd = -1;
    int32_t ret;
    if (innerUId->addr.type == ADDR_IPv4) {
        ret = bind_tcp_port_v4(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    } else {
        ret = bind_tcp_port_v6(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    }
    SHM_LOG_ERROR("init with env port fialed " << envPort << ", ret=" << ret);
    return ret;
}

int32_t parse_interface_with_type(const char *ipInfo, char *IP, sa_family_t &sockType, bool &flag)
{
    const char *delim = ":";
    const char *sep = strchr(ipInfo, delim[0]);
    if (sep != nullptr) {
        size_t leftLen = sep - ipInfo;
        if (leftLen >= MAX_IFCONFIG_LENGTH - 1 || leftLen == 0) {
            return ACLSHMEM_INVALID_VALUE;
        }
        strncpy(IP, ipInfo, leftLen);
        IP[leftLen] = '\0';
        sockType = (strcmp(sep + 1, "inet6") != 0) ? AF_INET : AF_INET6;
        flag = true;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_auto_get_ip(struct sockaddr *ifaAddr, char *local, sa_family_t &sockType)
{
    sockType = ifaAddr->sa_family;
    if (sockType == AF_INET) {
        auto localIp = reinterpret_cast<struct sockaddr_in *>(ifaAddr)->sin_addr;
        if (inet_ntop(sockType, &localIp, local, MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv4 to string failed. ");
            return ACLSHMEM_INVALID_PARAM;
        }
        return ACLSHMEM_SUCCESS;
    } else if (sockType == AF_INET6) {
        auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifaAddr)->sin6_addr;
        if (inet_ntop(sockType, &localIp, local, MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv6 to string failed. ");
            return ACLSHMEM_INVALID_PARAM;
        }
        return ACLSHMEM_SUCCESS;
    }
    return ACLSHMEM_INVALID_PARAM;
}

bool aclshmemi_check_ifa(struct ifaddrs *ifa, sa_family_t sockType, bool flag, char *ifaName, size_t ifaLen)
{
    if (ifa->ifa_addr == nullptr || ifa->ifa_netmask == nullptr || ifa->ifa_name == nullptr) {
        SHM_LOG_DEBUG("loop ifa_addr/ifa_netmask/ifa_name is nullptr");
        return false;
    }

    // socket type match and input env ifa valid
    if (ifa->ifa_addr->sa_family != sockType && flag) {
        SHM_LOG_DEBUG("sa family is not match, get " << ifa->ifa_addr->sa_family << ", expect " << sockType);
        return false;
    }

    //  prefix match with input ifa name
    if (strncmp(ifa->ifa_name, ifaName, ifaLen) != 0) {
        SHM_LOG_DEBUG("ifa name prefix un-match, get " << ifa->ifa_name << ", expect " << ifaName);
        return false;
    }

    // ignore ifa which is down or loopback or not running
    if ((ifa->ifa_flags & IFF_LOOPBACK) || !(ifa->ifa_flags & IFF_RUNNING) || !(ifa->ifa_flags & IFF_UP)) {
        SHM_LOG_DEBUG("ifa flag un-match, flag=" << ifa->ifa_flags);
        return false;
    }

    if (sockType == AF_INET6) {
        struct sockaddr_in6 *sa6 = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
        if (IN6_IS_ADDR_LINKLOCAL(&sa6->sin6_addr)) {
            SHM_LOG_DEBUG("ifa is scope link addr " << ifaName);
            return false;
        }
    }
    return true;
}

int32_t aclshmemi_get_ip_from_ifa(char *local, sa_family_t &sockType, const char *ipInfo)
{
    struct ifaddrs *ifaddr;
    char ifaName[MAX_IFCONFIG_LENGTH];
    sockType = AF_INET;
    bool flag = false;
    if (ipInfo == nullptr) {
        strncpy(ifaName, "eth", DEFAULT_IFNAME_LNEGTH);
        ifaName[DEFAULT_IFNAME_LNEGTH - 1] = '\0';
        SHM_LOG_INFO("use default if to find IP:" << ifaName);
    } else if (parse_interface_with_type(ipInfo, ifaName, sockType, flag) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("IP size set in SHMEM_CONF_STORE_MASTER_IF format has wrong length");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno);
        return ACLSHMEM_INVALID_PARAM;
    }
    int32_t result = ACLSHMEM_INVALID_PARAM;
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!aclshmemi_check_ifa(ifa, sockType, flag, ifaName, strlen(ifaName))) {
            continue;
        }
        if (sockType == AF_INET && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr)->sin_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv4 to string failed. ");
                continue;
            }
            result = ACLSHMEM_SUCCESS;
            break;
        } else if (sockType == AF_INET6 && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr)->sin6_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv6 to string failed. ");
                continue;
            }
            result = ACLSHMEM_SUCCESS;
            break;
        } else {
            auto ret = aclshmemi_auto_get_ip(ifa->ifa_addr, local, sockType);
            if (ret != ACLSHMEM_SUCCESS) {
                continue;
            }
            result = ACLSHMEM_SUCCESS;
            break;
        }
    }
    freeifaddrs(ifaddr);
    return result;
}

int32_t aclshmemi_get_ip_from_env(char *ip, uint16_t &port, sa_family_t &sockType, const char *ipPort)
{
    if (ipPort != nullptr) {
        SHM_LOG_DEBUG("get env SHMEM_UID_SESSION_ID value:" << ipPort);
        std::string ipPortStr = ipPort;

        if (ipPort[0] == '[') {
            sockType = AF_INET6;
            size_t found = ipPortStr.find_last_of(']');
            if (found == std::string::npos || ipPortStr.length() - found <= 1) {
                SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
                return ACLSHMEM_INVALID_PARAM;
            }
            std::string ipStr = ipPortStr.substr(1, found - 1);
            std::string portStr = ipPortStr.substr(found + 2);

            std::snprintf(ip, MAX_IP, "%s", ipStr.c_str());

            port = std::stoi(portStr);
        } else {
            sockType = AF_INET;
            size_t found = ipPortStr.find_last_of(':');
            if (found == std::string::npos || ipPortStr.length() - found <= 1) {
                SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
                return ACLSHMEM_INVALID_PARAM;
            }
            std::string ipStr = ipPortStr.substr(0, found);
            std::string portStr = ipPortStr.substr(found + 1);

            std::snprintf(ip, MAX_IP, "%s", ipStr.c_str());

            port = std::stoi(portStr);
        }
        return ACLSHMEM_SUCCESS;
    }
    return ACLSHMEM_INVALID_PARAM;
}

int32_t aclshmemi_set_ip_info(aclshmemx_uniqueid_t *uid, sa_family_t &sockType, char *pta_env_ip, uint16_t pta_env_port,
                          bool is_from_ifa)
{
    // init default uid
    SHM_ASSERT_RETURN(uid != nullptr, ACLSHMEM_INVALID_PARAM);
    *uid = ACLSHMEM_UNIQUEID_INITIALIZER;
    shmemx_bootstrap_uid_state_t *innerUID = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(uid);
    if (sockType == AF_INET) {
        innerUID->addr.addr.addr4.sin_family = AF_INET;
        if (inet_pton(AF_INET, pta_env_ip, &(innerUID->addr.addr.addr4.sin_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv4 failed");
            return ACLSHMEM_NOT_INITED;
        }
        innerUID->addr.type = ADDR_IPv4;
    } else if (sockType == AF_INET6) {
        innerUID->addr.addr.addr6.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, pta_env_ip, &(innerUID->addr.addr.addr6.sin6_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv6 failed");
            return ACLSHMEM_NOT_INITED;
        }
        innerUID->addr.type = ADDR_IPv6;
    } else {
        SHM_LOG_ERROR("IP Type is not IPv4 or IPv6");
        return ACLSHMEM_INVALID_PARAM;
    }

    // fill ip port as part of uid
    if (is_from_ifa) {
        int32_t ret = aclshmemi_get_port_magic(innerUID, pta_env_ip);
        if (ret != 0) {
            SHM_LOG_ERROR("get available port failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        int32_t ret = aclshmemi_using_env_port(innerUID, pta_env_ip, pta_env_port);
        if (ret != 0) {
            SHM_LOG_ERROR("using env port failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    SHM_LOG_INFO("gen unique id success.");
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_get_uniqueid_hybm(aclshmemx_uniqueid_t *uid)
{
    char pta_env_ip[MAX_IP];
    uint16_t pta_env_port;
    sa_family_t sockType;
    const char *ipPort = std::getenv("SHMEM_UID_SESSION_ID");
    const char *ipInfo = std::getenv("SHMEM_UID_SOCK_IFNAM");
    bool is_from_ifa = false;
    if (ipPort != nullptr) {
        if (aclshmemi_get_ip_from_env(pta_env_ip, pta_env_port, sockType, ipPort) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get pta master addr.");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        is_from_ifa = true;
        if (aclshmemi_get_ip_from_ifa(pta_env_ip, sockType, ipInfo) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get available ip port.");
            return ACLSHMEM_INVALID_PARAM;
        }
    }
    SHM_LOG_INFO("get master IP value:" << pta_env_ip);
    return aclshmemi_set_ip_info(uid, sockType, pta_env_ip, pta_env_port, is_from_ifa);
}

#endif