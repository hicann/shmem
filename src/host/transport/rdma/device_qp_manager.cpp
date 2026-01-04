/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dl_hccp_api.h"

#include "device_qp_manager.h"
using namespace shm;

DeviceQpManager::DeviceQpManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, sockaddr_in devNet,
                                 hybm_role_type role) noexcept
    : deviceId_{deviceId},
      rankId_{rankId},
      rankCount_{rankCount},
      deviceAddress_{devNet}
{
}

void *DeviceQpManager::CreateLocalSocket() noexcept
{
    void *socketHandle = nullptr;
    HccpRdev rdev;
    rdev.phyId = deviceId_;
    rdev.family = AF_INET;
    rdev.localIp.addr = deviceAddress_.sin_addr;
    auto ret = DlHccpApi::RaSocketInit(HccpNetworkMode::NETWORK_OFFLINE, rdev, socketHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("initialize socket handle failed: " << ret);
        return nullptr;
    }

    return socketHandle;
}

int DeviceQpManager::CreateServerSocket() noexcept
{
    if (serverSocketHandle_ != nullptr) {
        return ACLSHMEM_SUCCESS;
    }

    auto socketHandle = CreateLocalSocket();
    if (socketHandle == nullptr) {
        SHM_LOG_ERROR("create local socket handle failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    HccpSocketListenInfo listenInfo{};
    listenInfo.handle = socketHandle;
    listenInfo.port = deviceAddress_.sin_port;
    bool successListen = false;
    while (listenInfo.port <= std::numeric_limits<uint16_t>::max()) {
        auto ret = DlHccpApi::RaSocketListenStart(&listenInfo, 1);
        if (ret == 0) {
            deviceAddress_.sin_port = listenInfo.port;
            successListen = true;
            break;
        }
        listenInfo.port++;
    }
    if (!successListen) {
        SHM_LOG_ERROR("start to listen server socket failed.");
        DlHccpApi::RaSocketDeinit(socketHandle);
        return ACLSHMEM_INNER_ERROR;
    }

    SHM_LOG_INFO("start to listen on port: " << listenInfo.port << " success.");
    serverSocketHandle_ = socketHandle;
    return ACLSHMEM_SUCCESS;
}

void DeviceQpManager::DestroyServerSocket() noexcept
{
    if (serverSocketHandle_ == nullptr) {
        return;
    }

    HccpSocketListenInfo listenInfo{};
    listenInfo.handle = serverSocketHandle_;
    listenInfo.port = deviceAddress_.sin_port;
    auto ret = DlHccpApi::RaSocketListenStop(&listenInfo, 1);
    if (ret != 0) {
        SHM_LOG_INFO("stop to listen on port: " << listenInfo.port << " return: " << ret);
    }

    ret = DlHccpApi::RaSocketDeinit(serverSocketHandle_);
    if (ret != 0) {
        SHM_LOG_INFO("deinit server socket return: " << ret);
    }
    serverSocketHandle_ = nullptr;
}

static constexpr uint32_t SEND_CQ_DEPTH = 8192;
static constexpr uint32_t RECV_CQ_DEPTH = 128;
static constexpr uint32_t MAX_SEND_WR = 8192;
static constexpr uint32_t MAX_RECV_WR = 128;
static constexpr uint32_t MAX_SEND_SGE = 1;
static constexpr uint32_t MAX_RECV_SGE = 1;
static constexpr uint32_t RA_MAX_BATCH_NUM = 16;
static constexpr uint32_t QP_MODE = 2;
static constexpr uint32_t CALLER_POLL_CQ_CSTM = 1;

DeviceQpManager::~DeviceQpManager() noexcept
{
    CloseServices();
}

int DeviceQpManager::SetRemoteRankInfo(const std::unordered_map<uint32_t, ConnectRankInfo> &ranks) noexcept
{
    if (started_) {
        SHM_LOG_ERROR("fixed ranks not support update ranks info after startup");
        return ACLSHMEM_INNER_ERROR;
    }

    currentRanksInfo_ = ranks;
    return ACLSHMEM_SUCCESS;
}

int DeviceQpManager::SetLocalMemories(const MemoryRegionMap &mrs) noexcept
{
    if (started_) {
        SHM_LOG_INFO("fixed ranks not support update register MRs after startup");
        return ACLSHMEM_SUCCESS;
    }

    currentLocalMrs_ = mrs;
    return ACLSHMEM_SUCCESS;
}

int DeviceQpManager::Startup(void *rdma) noexcept
{
    if (rdma == nullptr) {
        SHM_LOG_ERROR("input rdma is null");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (started_) {
        SHM_LOG_ERROR("already started.");
        return ACLSHMEM_INNER_ERROR;
    }

    rdmaHandle_ = rdma;
    if (!ReserveQpInfoSpace()) {
        SHM_LOG_ERROR("reserve qp info space failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    if (currentRanksInfo_.size() != rankCount_) {
        SHM_LOG_ERROR("set rank count = " << currentRanksInfo_.size() << ", but rank_size = " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR("input options of nics contains rankId:" << it->first << ", rank count: " << rankCount_);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    auto ret = StartServerSide();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("start server side failed: " << ret);
        return ret;
    }

    ret = StartClientSide();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("start client side failed: " << ret);
        return ret;
    }

    started_ = true;
    return ACLSHMEM_SUCCESS;
}

void DeviceQpManager::Shutdown() noexcept
{
    CloseServices();
}

int DeviceQpManager::WaitingConnectionReady() noexcept
{
    if (serverConnectThread_ != nullptr) {
        serverConnectThread_->join();
        serverConnectThread_ = nullptr;
    }

    if (clientConnectThread_ != nullptr) {
        clientConnectThread_->join();
        clientConnectThread_ = nullptr;
    }

    if (serverConnectResult == ACLSHMEM_SUCCESS && clientConnectResult == ACLSHMEM_SUCCESS) {
        SHM_LOG_INFO("client & server connections ready.");
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_ERROR("background connection thread not started.");
    return ACLSHMEM_INNER_ERROR;
}

void *DeviceQpManager::GetQpInfoAddress() const noexcept
{
    return qpInfo_;
}

void *DeviceQpManager::GetQpHandleWithRankId(uint32_t rankId) const noexcept
{
    auto connections = rankId < rankId_ ? &clientConnections_ : &serverConnections_;
    auto pos = connections->find(rankId);
    if (pos == connections->end()) {
        return nullptr;
    }

    return pos->second.qpHandles[CONN_QP_STARS];
}

bool DeviceQpManager::ReserveQpInfoSpace() noexcept
{
    if (qpInfo_ != nullptr) {
        return true;
    }

    void *ptr = nullptr;
    auto oneQpSize = 2U * (sizeof(AiQpRMAWQ) + sizeof(AiQpRMACQ)) + sizeof(RdmaMemRegionInfo);
    qpInfoSize_ = sizeof(AiQpRMAQueueInfo) + oneQpSize * rankCount_;
    auto ret = aclrtMalloc(&ptr, qpInfoSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0) {
        SHM_LOG_ERROR("allocate device size: " << qpInfoSize_ << ", failed: " << ret);
        return false;
    }

    qpInfo_ = (AiQpRMAQueueInfo *)ptr;
    return true;
}

int DeviceQpManager::StartServerSide() noexcept
{
    if (rankId_ + 1U == rankCount_) {
        serverConnectResult = 0;
        return ACLSHMEM_SUCCESS;
    }

    auto ret = CreateServerSocket();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("create server socket failed: " << ret);
        return ret;
    }

    ret = GenerateWhiteList();
    if (ret != 0) {
        SHM_LOG_ERROR("generate white list failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    try {
        serverConnectThread_ = std::make_shared<std::thread>([this]() {
            aclrtSetDevice(deviceId_);
            serverConnectResult = WaitConnectionsReady(serverConnections_);
            if (serverConnectResult != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("[Rank " << rankId_ << "] wait connection ready failed: " << serverConnectResult);
            }
            
            serverConnectResult = CreateQpWaitingReady(serverConnections_, CONN_QP_AI_CORE);
            if (serverConnectResult != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("[Rank " << rankId_ << "] wait connection AI qp ready failed: " << serverConnectResult);
            }
            
            serverConnectResult = CreateQpWaitingReady(serverConnections_, CONN_QP_STARS);
            if (serverConnectResult != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("[Rank " << rankId_ << "] wait connection STARS qp ready failed: " << serverConnectResult);
            }
        });
    } catch (const std::exception& e) {
        SHM_LOG_ERROR("[Rank " << rankId_ << "] Failed to create server thread for rank " << rankId_ << ": " << e.what());
        return ACLSHMEM_INNER_ERROR;
    } catch (...) {
        SHM_LOG_ERROR("[Rank " << rankId_ << "] Unknown exception when creating server thread for rank " << rankId_);
        return ACLSHMEM_INNER_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

int DeviceQpManager::StartClientSide() noexcept
{
    if (rankId_ == 0U) {
        SHM_LOG_INFO("rankId: " << rankId_ << " need not connect to others.");
        clientConnectResult = ACLSHMEM_SUCCESS;
        return ACLSHMEM_SUCCESS;
    }

    std::vector<HccpSocketConnectInfo> connectInfos;
    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        if (it->first >= rankId_) {
            continue;  // client connect to small ranks.
        }

        auto socketHandle = CreateLocalSocket();
        if (socketHandle == nullptr) {
            SHM_LOG_ERROR("create local socket handle failed");
            CloseClientConnections();
            return ACLSHMEM_INNER_ERROR;
        }

        clientConnections_.emplace(it->first, ConnectionChannel{it->second.network.sin_addr, socketHandle});
        HccpSocketConnectInfo connectInfo;
        connectInfo.handle = socketHandle;
        connectInfo.remoteIp.addr = it->second.network.sin_addr;
        connectInfo.port = it->second.network.sin_port;
        bzero(connectInfo.tag, sizeof(connectInfo.tag));
        SHM_LOG_DEBUG("add connecting server " << connectInfo);
        connectInfos.emplace_back(connectInfo);
    }

    for (size_t i = 0; i < connectInfos.size(); i += RA_MAX_BATCH_NUM) {
        size_t currentBatchSize = (connectInfos.size() - i) >= RA_MAX_BATCH_NUM ? RA_MAX_BATCH_NUM : (connectInfos.size() - i);
        auto batchStart = connectInfos.begin() + i;
        auto batchEnd = batchStart + currentBatchSize;
        std::vector<HccpSocketConnectInfo> currentBatch(batchStart, batchEnd);

        auto ret = DlHccpApi::RaSocketBatchConnect(currentBatch.data(), currentBatch.size());
        if (ret != 0) {
            SHM_LOG_ERROR("[Rank " << rankId_ << "] connect to all servers failed: " << ret << ", servers count = " << connectInfos.size());
            CloseClientConnections();
            return ACLSHMEM_INNER_ERROR;
        }
    }

    try {
        clientConnectThread_ = std::make_shared<std::thread>([this]() {
            aclrtSetDevice(deviceId_);
            clientConnectResult = WaitConnectionsReady(clientConnections_);
            if (clientConnectResult != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("[Rank " << rankId_ << "] client wait connections failed: " << clientConnectResult);
                CloseClientConnections();
            }

            clientConnectResult = CreateQpWaitingReady(clientConnections_, CONN_QP_AI_CORE);
            if (clientConnectResult != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("[Rank " << rankId_ << "] client create qp for AI CORE failed: " << clientConnectResult);
                CloseClientConnections();
            }

            clientConnectResult = CreateQpWaitingReady(clientConnections_, CONN_QP_STARS);
            if (clientConnectResult != ACLSHMEM_SUCCESS) {
                SHM_LOG_ERROR("[Rank " << rankId_ << "] client create qp for STARS failed: " << clientConnectResult);
                CloseClientConnections();
            }
        });
    } catch (const std::exception &e) {
        SHM_LOG_ERROR("[Rank " << rankId_ << "]Failed to create client thread: " << e.what());
        return ACLSHMEM_INNER_ERROR;
    } catch (...) {
        SHM_LOG_ERROR("[Rank " << rankId_ << "] Unknown exception when creating client thread for rank " << rankId_);
        return ACLSHMEM_INNER_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

int DeviceQpManager::GenerateWhiteList() noexcept
{
    std::vector<HccpSocketWhiteListInfo> whitelist;
    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        if (it->first <= rankId_) {
            continue;  // small id as server, large id as client
        }
        HccpSocketWhiteListInfo info{};
        info.remoteIp.addr = it->second.network.sin_addr;
        info.connLimit = rankCount_;
        bzero(info.tag, sizeof(info.tag));
        whitelist.emplace_back(info);
        serverConnections_.emplace(it->first, ConnectionChannel{info.remoteIp.addr, serverSocketHandle_});
    }

    if (whitelist.empty()) {
        return ACLSHMEM_SUCCESS;
    }

    for (size_t i = 0; i < whitelist.size(); i += RA_MAX_BATCH_NUM) {
        size_t currentBatchSize = (whitelist.size() - i) >= RA_MAX_BATCH_NUM ? RA_MAX_BATCH_NUM : (whitelist.size() - i);
        auto batchStart = whitelist.begin() + i;
        auto batchEnd = batchStart + currentBatchSize;
        std::vector<HccpSocketWhiteListInfo> currentBatch(batchStart, batchEnd);
        auto ret = DlHccpApi::RaSocketWhiteListAdd(serverSocketHandle_, currentBatch.data(), currentBatch.size());
        if (ret != 0) {
            SHM_LOG_ERROR("[Rank " << rankId_ << "] socket handle add white list failed: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }

    return ACLSHMEM_SUCCESS;
}

int DeviceQpManager::WaitConnectionsReady(std::unordered_map<uint32_t, ConnectionChannel> &connections) noexcept
{
    uint32_t cnt = 0;
    auto start = std::chrono::steady_clock::now();
    auto timeout = start + std::chrono::minutes(2);
    SHM_LOG_DEBUG(rankId_ << " begin wait connections, size=" << connections.size());

    std::vector<HccpSocketInfo> socketInfos;
    std::unordered_map<in_addr_t, uint32_t> addr2index;
    for (auto it = connections.begin(); it != connections.end(); ++it) {
        if (it->second.socketFd != nullptr) {
            continue;
        }

        HccpSocketInfo info{};
        info.handle = it->second.socketHandle;
        info.fd = nullptr;
        info.remoteIp.addr = it->second.remoteIp;
        info.status = 0;
        bzero(info.tag, sizeof(info.tag));
        socketInfos.push_back(info);
        addr2index.emplace(it->second.remoteIp.s_addr, it->first);
    }

    if (socketInfos.empty()) {
        SHM_LOG_WARN(rankId_ << " socketInfos is empty.");
        return ACLSHMEM_SUCCESS;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto role = (&connections == &clientConnections_) ? 1 : 0;

    do {
        uint32_t getSize = socketInfos.size() < RA_MAX_BATCH_NUM ? socketInfos.size() : RA_MAX_BATCH_NUM;
        auto ret = DlHccpApi::RaGetSockets(role, socketInfos.data(), getSize, cnt);
        if (ret != 0) {
            SHM_LOG_ERROR(rankId_ << " role(" << role << ") side get sockets failed: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100L));
        if (cnt == 0) {
            continue;
        }
        for (auto i = 0U; i < getSize; i++) {
            if (socketInfos[i].status != 1) {
                continue;
            }
            in_addr_t addr = socketInfos[i].remoteIp.addr.s_addr;
            char ipStr[INET6_ADDRSTRLEN];
            char* result {};
            result = inet_ntoa(socketInfos[i].remoteIp.addr);
            auto socketInfoPos = addr2index.find(addr);
            if (socketInfoPos == addr2index.end()) {
                SHM_LOG_ERROR(rankId_ << " socket ip(" << result << ") should not exist.");
                return ACLSHMEM_INNER_ERROR;
            }

            auto rankId = socketInfoPos->second;
            auto pos = connections.find(rankId);
            if (pos == connections.end()) {
                SHM_LOG_ERROR(rankId_ << " -> " << rankId << ":socket ip(" << result << ") should not exist.");
                return ACLSHMEM_INNER_ERROR;
            }

            if (pos->second.socketFd != nullptr) {
                SHM_LOG_ERROR(rankId_ << " -> " << rankId << ":get socket ip(" << result << ") already get socket fd.");
                return ACLSHMEM_INNER_ERROR;
            }

            if (pos->second.socketHandle != socketInfos[i].handle) {
                SHM_LOG_ERROR(rankId_ << " -> " << rankId << ":get socket ip(" << result << ") socket handle not match.");
                return ACLSHMEM_INNER_ERROR;
            }

            pos->second.socketFd = socketInfos[i].fd;
            SHM_LOG_INFO(rankId_ << " connect to (" << rankId << ") ready.");
        }
        std::vector<HccpSocketInfo>::iterator it = socketInfos.begin();
        for (; it != socketInfos.end();) {
            if (it->status == 1) {
                it = socketInfos.erase(it);
            } else {
                it++;
            }
        }
    } while (socketInfos.size() > 0);
    return ACLSHMEM_SUCCESS;
}
int DeviceQpManager::CreateQpWaitingReady(std::unordered_map<uint32_t, ConnectionChannel> &connections,
                                              ConnQpType qpType) noexcept
{
    for (auto it = connections.begin(); it != connections.end(); ++it) {
        auto ret = CreateOneQp(qpType, it->second);
        if (ret != 0) {
            SHM_LOG_ERROR("create QP type:" << qpType << " to " << it->first << " failed: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }

        for (auto pos = currentLocalMrs_.begin(); pos != currentLocalMrs_.end(); ++pos) {
            HccpMrInfo info{};
            info.addr = (void *)(ptrdiff_t)pos->second.address;
            info.size = pos->second.size;
            info.access = 7;
            ret = DlHccpApi::RaMrReg(it->second.qpHandles[qpType], info);
            if (ret != 0) {
                SHM_LOG_ERROR("register MR failed: " << ret);
                return ACLSHMEM_INNER_ERROR;
            }
        }

        ret = DlHccpApi::RaQpConnectAsync(it->second.qpHandles[qpType], it->second.socketFd);
        if (ret != 0) {
            SHM_LOG_ERROR("connect AI QP to " << it->first << " failed: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }

    auto start = std::chrono::steady_clock::now();
    auto timeout = start + std::chrono::minutes(1);
    while (std::chrono::steady_clock::now() < timeout) {
        int connectingCount = 0;
        for (auto it = connections.begin(); it != connections.end(); ++it) {
            int status = 0;
            auto ret = DlHccpApi::RaGetQpStatus(it->second.qpHandles[qpType], status);
            if (ret != 0) {
                SHM_LOG_ERROR("get AI QP status to " << it->first << " failed: " << ret);
                return ACLSHMEM_INNER_ERROR;
            }
            if (status != 1) {
                connectingCount++;
            }
        }
        if (connectingCount == 0) {
            return FillQpInfo(qpType);
        }
    }
    return ACLSHMEM_INNER_ERROR;
}

int DeviceQpManager::CreateOneQp(ConnQpType qpType, ConnectionChannel &channel) noexcept
{
    int ret;
    if (qpType == CONN_QP_AI_CORE) {
        HccpQpExtAttrs attr{};
        attr.qpMode = NETWORK_OFFLINE;
        attr.version = 1;
        attr.cqAttr.sendCqDepth = SEND_CQ_DEPTH;
        attr.cqAttr.recvDqDepth = RECV_CQ_DEPTH;
        attr.qp_attr.cap.max_send_wr = MAX_SEND_WR;
        attr.qp_attr.cap.max_send_sge = MAX_SEND_SGE;
        attr.qp_attr.cap.max_recv_wr = MAX_RECV_WR;
        attr.qp_attr.cap.max_recv_sge = MAX_RECV_SGE;
        attr.qp_attr.qp_type = IBV_QPT_RC;
        attr.data_plane_flag.bs.cq_cstm = CALLER_POLL_CQ_CSTM;
        ret = DlHccpApi::RaQpAiCreate(rdmaHandle_, attr, channel.aiQpInfo, channel.qpHandles[qpType]);
    } else {
        ret = DlHccpApi::RaQpCreate(rdmaHandle_, 0, QP_MODE, channel.qpHandles[qpType]);
    }
    return ret;
}

int DeviceQpManager::FillQpInfo(ConnQpType qpType) noexcept
{
    if (qpType != CONN_QP_AI_CORE) {
        return ACLSHMEM_SUCCESS;
    }

    const uint32_t slevel = 4;
    std::vector<uint8_t> qpInfoBuffer(qpInfoSize_);
    auto copyInfo = (AiQpRMAQueueInfo *)(void *)qpInfoBuffer.data();
    copyInfo->count = 1;
    copyInfo->sq = (AiQpRMAWQ *)(void *)(copyInfo + 1);
    copyInfo->rq = (AiQpRMAWQ *)(void *)(copyInfo->sq + rankCount_);
    copyInfo->scq = (AiQpRMACQ *)(void *)(copyInfo->rq + rankCount_);
    copyInfo->rcq = (AiQpRMACQ *)(void *)(copyInfo->scq + rankCount_);
    copyInfo->mr = (RdmaMemRegionInfo *)(void *)(copyInfo->rcq + rankCount_);
    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        copyInfo->mr[it->first].size = it->second.mr.size;
        copyInfo->mr[it->first].addr = it->second.mr.address;
        copyInfo->mr[it->first].lkey = it->second.mr.lkey;
        copyInfo->mr[it->first].rkey = it->second.mr.rkey;
        if (it->first == rankId_) {
            continue;
        }

        std::unordered_map<uint32_t, ConnectionChannel> *connections;
        if (it->first < rankId_) {
            connections = &clientConnections_;
        } else {
            connections = &serverConnections_;
        }

        auto pos = connections->find(it->first);
        if (pos == connections->end()) {
            SHM_LOG_ERROR("missing for remote: " << it->first);
            return ACLSHMEM_INNER_ERROR;
        }

        CopyAiWQInfo(copyInfo->sq[it->first], pos->second.aiQpInfo.data_plane_info.sq, DBMode::HW_DB, slevel);
        CopyAiWQInfo(copyInfo->rq[it->first], pos->second.aiQpInfo.data_plane_info.rq, DBMode::SW_DB, slevel);
        CopyAiCQInfo(copyInfo->scq[it->first], pos->second.aiQpInfo.data_plane_info.scq, DBMode::HW_DB);
        CopyAiCQInfo(copyInfo->rcq[it->first], pos->second.aiQpInfo.data_plane_info.rcq, DBMode::SW_DB);
    }

    auto pointer = (ptrdiff_t)(void *)(qpInfo_);
    pointer += sizeof(AiQpRMAQueueInfo);
    copyInfo->sq = (AiQpRMAWQ *)(void *)(pointer);

    pointer += sizeof(AiQpRMAWQ) * rankCount_;
    copyInfo->rq = (AiQpRMAWQ *)(void *)(pointer);

    pointer += sizeof(AiQpRMAWQ) * rankCount_;
    copyInfo->scq = (AiQpRMACQ *)(void *)(pointer);

    pointer += sizeof(AiQpRMACQ) * rankCount_;
    copyInfo->rcq = (AiQpRMACQ *)(void *)(pointer);

    pointer += sizeof(AiQpRMACQ) * rankCount_;
    copyInfo->mr = (RdmaMemRegionInfo *)(void *)pointer;

    auto ret = aclrtMemcpy(qpInfo_, qpInfoSize_, copyInfo, qpInfoSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("copy qp info to device failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("copy qp info success");

    return ACLSHMEM_SUCCESS;
}

void DeviceQpManager::CopyAiWQInfo(struct AiQpRMAWQ &dest, const struct ai_data_plane_wq &src, DBMode dbMode,
                                       uint32_t sl) noexcept
{
    dest.wqn = src.wqn;
    dest.bufAddr = src.buf_addr;
    dest.wqeSize = src.wqebb_size;
    dest.depth = src.depth;
    dest.headAddr = src.head_addr;
    dest.tailAddr = src.tail_addr;
    dest.dbMode = dbMode;
    if (dbMode == DBMode::SW_DB) {
        dest.dbAddr = src.swdb_addr;
    } else if (dbMode == DBMode::HW_DB) {
        dest.dbAddr = src.db_reg;
    }
    dest.sl = sl;
    SHM_LOG_INFO("CopyAiWQInfo: wqn = " << dest.wqn << ", bufAddr = " << dest.bufAddr << ", wqeSize = "
                    << dest.wqeSize << ", depth = " << dest.depth << ", headAddr = " << dest.headAddr
                    << ", tailAddr = " << dest.tailAddr << ", dbAddr = " << dest.dbAddr
                    << ", sl = " << dest.sl);
}

void DeviceQpManager::CopyAiCQInfo(struct AiQpRMACQ &dest, const ai_data_plane_cq &source, DBMode dbMode) noexcept
{
    dest.cqn = source.cqn;
    dest.bufAddr = source.buf_addr;
    dest.cqeSize = source.cqe_size;
    dest.depth = source.depth;
    dest.headAddr = source.head_addr;
    dest.tailAddr = source.tail_addr;
    dest.dbMode = dbMode;
    if (dbMode == DBMode::SW_DB) {
        dest.dbAddr = source.swdb_addr;
    } else if (dbMode == DBMode::HW_DB) {
        dest.dbAddr = source.db_reg;
    }
    SHM_LOG_INFO("CopyAiCQInfo: cqn = " << dest.cqn << ", bufAddr = " << dest.bufAddr << ", cqeSize = "
                    << dest.cqeSize << ", depth = " << dest.depth << ", headAddr = " << dest.headAddr
                    << ", tailAddr = " << dest.tailAddr << ", dbAddr = " << dest.dbAddr);
}

void DeviceQpManager::CloseServices() noexcept
{
    CloseServerConnections();
    CloseClientConnections();
}

void DeviceQpManager::CloseClientConnections() noexcept
{
    CloseConnections(clientConnections_);
}

void DeviceQpManager::CloseServerConnections() noexcept
{
    DestroyServerSocket();
    CloseConnections(serverConnections_);
}

void DeviceQpManager::CloseConnections(std::unordered_map<uint32_t, ConnectionChannel> &connections) noexcept
{
    std::vector<HccpSocketCloseInfo> socketCloseInfos;
    for (auto it = connections.begin(); it != connections.end(); ++it) {
        if (it->second.qpHandles[CONN_QP_AI_CORE] != nullptr) {
            auto ret = DlHccpApi::RaQpDestroy(it->second.qpHandles[CONN_QP_AI_CORE]);
            if (ret != 0) {
                SHM_LOG_WARN("destroy AI QP to server: " << it->first << " failed: " << ret);
            }
            it->second.qpHandles[CONN_QP_AI_CORE] = nullptr;
        }

        if (it->second.qpHandles[CONN_QP_STARS] != nullptr) {
            auto ret = DlHccpApi::RaQpDestroy(it->second.qpHandles[CONN_QP_STARS]);
            if (ret != 0) {
                SHM_LOG_WARN("destroy stars QP to server: " << it->first << " failed: " << ret);
            }
            it->second.qpHandles[CONN_QP_STARS] = nullptr;
        }

        if (it->second.socketFd != nullptr) {
            HccpSocketCloseInfo info;
            info.handle = it->second.socketHandle;
            info.fd = it->second.socketFd;
            info.linger = 0;
            socketCloseInfos.push_back(info);
            it->second.socketFd = nullptr;
        }
    }

    if (!socketCloseInfos.empty()) {
        auto ret = DlHccpApi::RaSocketBatchClose(socketCloseInfos.data(), socketCloseInfos.size());
        if (ret != 0) {
            SHM_LOG_INFO("close sockets return: " << ret);
        }
    }

    for (auto it = connections.begin(); it != connections.end(); ++it) {
        auto ret = DlHccpApi::RaSocketDeinit(it->second.socketHandle);
        if (ret != 0) {
            SHM_LOG_INFO("deinit socket to server: " << it->first << " return: " << ret);
        }
    }

    connections.clear();
}