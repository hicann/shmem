/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RDMA_MANAGER_H
#define RDMA_MANAGER_H

#include <iostream>
#include <netinet/in.h>
#include <dlfcn.h>
#include "dl_hccp_api.h"
#include "acl/acl.h"
#include "device_qp_manager.h"

const char *g_rt_lib_name = "libascendcl.so";
int (*rtGetLogicDevIdByUserDevIdFunc)(const int32_t, int32_t *const);

class rdma_manager {
public:
    rdma_manager() {}

    ~rdma_manager() {
        delete qpManager_;
        ClearAllRegisterMRs();
        tsdOpened_ = false;
        raInitialized_ = false;
        deviceIpRetired_ = false;
        storedRdmaHandle_ = nullptr;
    }

    void ClearAllRegisterMRs()
    {
        for (auto it = registerMRS_.begin(); it != registerMRS_.end(); ++it) {
            auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, it->second.mrHandle);
            if (ret != 0) {
                SHM_LOG_ERROR("Unregister:" << (void *)(ptrdiff_t)it->first << " : " << it->second << " failed: " << ret);
            }
        }
        registerMRS_.clear();
    }

    int OpenDevice(const TransportOptions &options)
    {
        int32_t deviceId = options.dev_id;
        int32_t logicDeviceId = options.logic_dev_id;
        deviceId_ = static_cast<uint32_t>(logicDeviceId);
        rankId_ = options.rankId;
        rankCount_ = options.rankCount;
        role_ = options.role;
        auto port = options.nic;
        if (port < 0 || port > 65536) {
            SHM_LOG_ERROR("Failed to parse nic info, nic = " << options.nic);
        }
        devicePort_ = static_cast<uint16_t>(port);
        DlHccpApi::LoadLibrary();

        if (!PrepareOpenDevice(deviceId, rankCount_, deviceIp_, rdmaHandle_, deviceId_)) {
            SHM_LOG_ERROR("PrepareOpenDevice failed.");
            return -1;
        }

        SHM_LOG_INFO("ip = " << inet_ntoa(deviceIp_) << ", port = " << devicePort_);

        sockaddr_in deviceAddr;
        deviceAddr.sin_family = AF_INET;
        deviceAddr.sin_addr = deviceIp_;
        deviceAddr.sin_port = devicePort_;
        qpManager_ = new DeviceQpManager(deviceId_, rankId_, rankCount_, deviceAddr, HYBM_ROLE_PEER);

        return 0;
    }

    void* GetQPInfoAddr() {
        return qpManager_->GetQpInfoAddress();
    }

    in_addr GetDeviceIP() {
        return deviceIp_;
    }

    Result RegisterMemoryRegion(const TransportMemoryRegion &mr)
    {
        void *mrHandle = nullptr;
        HccpMrInfo info{};
        info.addr = (void *)(ptrdiff_t)mr.addr;
        info.size = mr.size;
        info.access = mr.access;
        auto ret = DlHccpApi::RaRegisterMR(rdmaHandle_, &info, mrHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("register MR=" << mr << " failed: " << ret);
            return SHMEM_INNER_ERROR;
        }

        RegMemResult result{mr.addr, mr.size, mrHandle, info.lkey, info.rkey};
        localMR_ = result;
        SHM_LOG_DEBUG("register MR result=" << result);

        registerMRS_.emplace(mr.addr, result);
        ret = qpManager_->SetLocalMemories(registerMRS_);
        if (ret != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("qp manager set mr failed: " << ret);
            return ret;
        }
        return 0;
    }

    Result UnregisterMemoryRegion(uint64_t addr)
    {
        auto pos = registerMRS_.find(addr);
        if (pos == registerMRS_.end()) {
            SHM_LOG_ERROR("input address not register!");
            return SHMEM_INVALID_PARAM;
        }

        auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, pos->second.mrHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("Unregister MR addr failed: " << ret);
            return SHMEM_INNER_ERROR;
        }

        registerMRS_.erase(pos);
        return 0;
    }

    RegMemResult GetLocalMR() {
        return localMR_;
    }

    Result Prepare(const HybmTransPrepareOptions &options)
    {
        SHM_LOG_DEBUG("RdmaTransportManager Prepare with : " << options);
        int ret;
        if ((ret = CheckPrepareOptions(options)) != 0) {
            return ret;
        }

        sockaddr_in deviceNetwork;
        std::unordered_map<uint32_t, ConnectRankInfo> rankInfo;
        for (auto it = options.options.begin(); it != options.options.end(); ++it) {
            ret = ipPortStringToSockaddr(it->second.nic, deviceNetwork);
            if (ret != SHMEM_SUCCESS) {
                SHM_LOG_ERROR("parse networks[" << it->first << "]=" << it->second.nic << " failed: " << ret);
                return SHMEM_INVALID_PARAM;
            }

            rankInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.mr});
        }

        ret = qpManager_->SetRemoteRankInfo(rankInfo);
        if (ret != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("qp manager set remote rank info failed: " << ret);
            return ret;
        }

        ret = qpManager_->Startup(rdmaHandle_);
        if (ret != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("qp manager startup failed: " << ret);
            return ret;
        }

        return SHMEM_SUCCESS;
    }

    Result Connect()
    {
        auto ret = AsyncConnect();
        if (ret != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("AsyncConnect() failed: " << ret);
            return ret;
        }

        ret = WaitForConnected(-1L);
        if (ret != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("WaitForConnected(-1) failed: " << ret);
            return ret;
        }

        return SHMEM_SUCCESS;
    }
private:
    bool OpenTsd(uint32_t deviceId, uint32_t rankCount)
    {
        if (tsdOpened_) {
            SHM_LOG_INFO("tsd already opened.");
            return true;
        }

        auto res = DlHccpApi::TsdOpen(deviceId, rankCount);
        if (res != 0) {
            SHM_LOG_ERROR("TsdOpen for (deviceId=" << deviceId << ", rankCount=" << rankCount << ") failed: " << res);
            return false;
        }

        SHM_LOG_DEBUG("open tsd for device id: " << deviceId << ", rank count: " << rankCount << " success.");
        tsdOpened_ = true;
        return true;
    }

    bool RaInit(uint32_t deviceId)
    {
        if (raInitialized_) {
            SHM_LOG_INFO("ra already initialized.");
            return true;
        }

        HccpRaInitConfig initConfig{};
        initConfig.phyId = deviceId;
        initConfig.nicPosition = NETWORK_OFFLINE;
        initConfig.hdcType = 6;  // HDC_SERVICE_TYPE_RDMA = 6
        SHM_LOG_DEBUG("RaInit=" << initConfig);
        auto ret = DlHccpApi::RaInit(initConfig);
        if (ret != 0) {
            SHM_LOG_ERROR("Hccp Init RA failed: " << ret);
            return false;
        }

        SHM_LOG_DEBUG("ra init for device id: " << deviceId << " success.");
        raInitialized_ = true;
        return true;
    }

    bool RetireDeviceIp(uint32_t deviceId, in_addr &deviceIp)
    {
        static in_addr retiredIp{};

        if (deviceIpRetired_) {
            SHM_LOG_INFO("device ip already retired : " << inet_ntoa(retiredIp));
            deviceIp = retiredIp;
            return true;
        }

        uint32_t count = 0;
        std::vector<HccpInterfaceInfo> infos;

        HccpRaGetIfAttr config;
        config.phyId = deviceId;
        config.nicPosition = NETWORK_OFFLINE;
        config.isAll = true;

        auto ret = DlHccpApi::RaGetIfNum(config, count);
        if (ret != 0 || count == 0) {
            SHM_LOG_ERROR("get interface count failed: " << ret << ", count: " << count);
            return false;
        }

        infos.resize(count);
        ret = DlHccpApi::RaGetIfAddrs(config, infos.data(), count);
        if (ret != 0) {
            SHM_LOG_ERROR("get interface information failed: " << ret);
            return false;
        }

        for (auto &info : infos) {
            if (info.family == AF_INET) {
                deviceIp = retiredIp = info.ifaddr.ip.addr;
                deviceIpRetired_ = true;
                SHM_LOG_DEBUG("retire device ip success : " << inet_ntoa(deviceIp));
                return true;
            }
        }

        SHM_LOG_ERROR("not found network device of AF_INET on NPU.");
        return false;
    }

    bool RaRdevInit(uint32_t deviceId, in_addr deviceIp, void *&rdmaHandle)
    {
        if (storedRdmaHandle_ != nullptr) {
            SHM_LOG_INFO("ra rdev already initialized.");
            rdmaHandle = storedRdmaHandle_;
            return true;
        }

        HccpRdevInitInfo info{};
        HccpRdev rdev{};

        info.mode = NETWORK_OFFLINE;
        info.notifyType = NOTIFY;
        info.enabled2mbLite = true;
        rdev.phyId = deviceId;
        rdev.family = AF_INET;
        rdev.localIp.addr = deviceIp;
        SHM_LOG_DEBUG("RaRdevInitV2, info=" << info << "rdev=" << rdev);
        auto ret = DlHccpApi::RaRdevInitV2(info, rdev, rdmaHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("Hccp Init RDev failed: " << ret);
            return false;
        }

        storedRdmaHandle_ = rdmaHandle;
        SHM_LOG_INFO("initialize RDev success.");
        return true;
    }

    bool PrepareOpenDevice(uint32_t device, uint32_t rankCount, in_addr &deviceIp, void *&rdmaHandle, uint32_t logicDeviceId)
    {
        // If can get rdmaHanle, maybe the device has beed opened, can try get rdmaHanle directly.
        if (DlHccpApi::RaRdevGetHandle(device, rdmaHandle) == 0) {
            if (rdmaHandle != nullptr) {
                if (!RetireDeviceIp(device, deviceIp)) {
                    SHM_LOG_ERROR("RetireDeviceIp failed.");
                    return false;
                }
                SHM_LOG_DEBUG("Had prepared device and get rdmaHandle success.");
                return true;
            }
            SHM_LOG_INFO("Had prepared device, but RdmaHadle is null, need init again.");
        }
        if (!OpenTsd(device, rankCount)) {
            SHM_LOG_ERROR("open tsd failed.");
            return false;
        }

        if (!RaInit(logicDeviceId)) {
            SHM_LOG_ERROR("RaInit failed.");
            return false;
        }

        if (!RetireDeviceIp(logicDeviceId, deviceIp)) {
            SHM_LOG_ERROR("RetireDeviceIp failed.");
            return false;
        }

        if (!RaRdevInit(logicDeviceId, deviceIp, rdmaHandle)) {
            SHM_LOG_ERROR("RaRdevInit failed.");
            return false;
        }
        return true;
    }

    Result AsyncConnect()
    {
        return SHMEM_SUCCESS;
    }

    Result WaitForConnected(int64_t timeoutNs)
    {
        if (qpManager_ == nullptr) {
            SHM_LOG_ERROR("server side not listen!");
            return SHMEM_INNER_ERROR;
        }

        auto ret = qpManager_->WaitingConnectionReady();
        if (ret != SHMEM_SUCCESS) {
            SHM_LOG_ERROR("wait for server side connected on device failed: " << ret);
            return ret;
        }

        return SHMEM_SUCCESS;
    }

    int CheckPrepareOptions(const HybmTransPrepareOptions &options)
    {
        if (role_ != HYBM_ROLE_PEER) {
            SHM_LOG_INFO("transport role: " << role_ << " check options passed.");
            return SHMEM_SUCCESS;
        }

        if (options.options.size() > rankCount_) {
            SHM_LOG_ERROR("options size():" << options.options.size() << " larger than rank count: " << rankCount_);
            return SHMEM_INVALID_PARAM;
        }

        if (options.options.find(rankId_) == options.options.end()) {
            SHM_LOG_ERROR("options not contains self rankId: " << rankId_);
            return SHMEM_INVALID_PARAM;
        }

        for (auto it = options.options.begin(); it != options.options.end(); ++it) {
            if (it->first >= rankCount_) {
                SHM_LOG_ERROR("input options of nics contains rankId:" << it->first << ", rank count: " << rankCount_);
                return SHMEM_INVALID_PARAM;
            }
        }

        return SHMEM_SUCCESS;
    }

    Result ipPortStringToSockaddr(const std::string& ip_port_str, sockaddr_in& addr) {
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;

        size_t colon_pos = ip_port_str.find(':');
        if (colon_pos == std::string::npos || 
            colon_pos == 0 || 
            colon_pos == ip_port_str.length() - 1) {
            SHM_LOG_ERROR("format mismatch");
            return SHMEM_INNER_ERROR;
        }

        std::string ip_str = ip_port_str.substr(0, colon_pos);
        std::string port_str = ip_port_str.substr(colon_pos + 1);

        if (port_str.empty()) {
            SHM_LOG_ERROR("Port not available!");
            return SHMEM_INNER_ERROR;
        }
        
        for (char c : port_str) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                SHM_LOG_ERROR("Port contains non-digit characters!");
                return SHMEM_INNER_ERROR;
            }
        }

        char* endptr;
        unsigned long port = std::strtoul(port_str.c_str(), &endptr, 10);
        
        if (endptr == port_str.c_str() || *endptr != '\0' || 
            port == 0 || port > 65535) {
            SHM_LOG_ERROR("Port out of range!");
            return SHMEM_INNER_ERROR;
        }
        
        addr.sin_port = htons(static_cast<uint16_t>(port));

        // Transform IP address
        if (inet_pton(AF_INET, ip_str.c_str(), &addr.sin_addr) != 1) {
            SHM_LOG_ERROR("IP address invalid!");
            return SHMEM_INNER_ERROR;
        }

        return SHMEM_SUCCESS;
    }

    uint32_t rankId_{0};
    uint32_t rankCount_{1};
    uint32_t deviceId_{0};
    hybm_role_type role_{HYBM_ROLE_PEER};
    in_addr deviceIp_{0};
    uint16_t devicePort_{0};
    void *rdmaHandle_{nullptr};
    void *storedRdmaHandle_{nullptr};
    bool tsdOpened_{0};
    bool raInitialized_{0};
    bool deviceIpRetired_{0};
    DeviceQpManager* qpManager_;
    RegMemResult localMR_;
    MemoryRegionMap registerMRS_;
};

#endif // RDMA_MANAGER_H