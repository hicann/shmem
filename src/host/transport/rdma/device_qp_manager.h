/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEVICE_QP_MANAGER_H
#define DEVICE_QP_MANAGER_H

#include <netinet/in.h>
#include <cstdint>
#include <unordered_map>
#include "dl_hccp_api.h"

class DeviceQpManager {
public:
    DeviceQpManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, sockaddr_in devNet,
                    hybm_role_type role) noexcept;
    ~DeviceQpManager() noexcept;

    int SetRemoteRankInfo(const std::unordered_map<uint32_t, ConnectRankInfo> &ranks) noexcept;
    int SetLocalMemories(const MemoryRegionMap &mrs) noexcept;
    int Startup(void *rdma) noexcept;
    void Shutdown() noexcept;
    int WaitingConnectionReady() noexcept;
    void *GetQpInfoAddress() const noexcept;
    void *GetQpHandleWithRankId(uint32_t rankId) const noexcept;

protected:
    void *CreateLocalSocket() noexcept;
    int CreateServerSocket() noexcept;
    void DestroyServerSocket() noexcept;

protected:
    const uint32_t deviceId_;
    const uint32_t rankId_;
    const uint32_t rankCount_;
    const hybm_role_type rankRole_;
    sockaddr_in deviceAddress_;
    void *serverSocketHandle_{nullptr};

private:
    enum ConnQpType : uint32_t {
        CONN_QP_AI_CORE,  // AI core使用的QP
        CONN_QP_STARS,    // Host侧使用STARS驱动的QP
        CONN_QP_COUNT
    };

    struct ConnectionChannel {
        in_addr remoteIp;
        void *socketHandle;
        void *socketFd{nullptr};
        void *qpHandles[CONN_QP_COUNT]{};
        HccpAiQpInfo aiQpInfo{};
        int qpStatus{-1};

        explicit ConnectionChannel(const in_addr ip) : ConnectionChannel{ip, nullptr} {}
        ConnectionChannel(in_addr ip, void *sock) : remoteIp{ip}, socketHandle{sock} {}
    };

    bool ReserveQpInfoSpace() noexcept;
    int StartServerSide() noexcept;
    int StartClientSide() noexcept;
    int GenerateWhiteList() noexcept;
    int WaitConnectionsReady(std::unordered_map<uint32_t, ConnectionChannel> &connections) noexcept;
    int CreateQpWaitingReady(std::unordered_map<uint32_t, ConnectionChannel> &connections, ConnQpType qpType) noexcept;
    int CreateOneQp(ConnQpType qpType, ConnectionChannel &channel) noexcept;
    int FillQpInfo(ConnQpType qpType) noexcept;
    void CopyAiWQInfo(struct AiQpRMAWQ &dest, const struct ai_data_plane_wq &src, DBMode dbMode, uint32_t sl) noexcept;
    void CopyAiCQInfo(struct AiQpRMACQ &dest, const ai_data_plane_cq &source, DBMode dbMode) noexcept;
    void CloseServices() noexcept;
    void CloseClientConnections() noexcept;
    void CloseServerConnections() noexcept;
    void CloseConnections(std::unordered_map<uint32_t, ConnectionChannel> &connections) noexcept;

    bool started_{false};
    int serverConnectResult{-1};
    int clientConnectResult{-1};
    uint32_t qpInfoSize_{0};
    void *rdmaHandle_{nullptr};
    std::unordered_map<uint32_t, ConnectRankInfo> currentRanksInfo_;
    MemoryRegionMap currentLocalMrs_;
    AiQpRMAQueueInfo *qpInfo_{nullptr};
    std::unordered_map<uint32_t, ConnectionChannel> clientConnections_;
    std::unordered_map<uint32_t, ConnectionChannel> serverConnections_;
};

#endif  // DEVICE_QP_MANAGER_H