/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dl_comm_def.h"
#include "dl_acl_api.h"
#include "shmemi_net_util.h"
#include "hybm_device_mem_segment.h"
#include "host/shmem_host_def.h"
#include "hybm_device_mem_segment.h"

namespace shm {
bool MemSegment::deviceInfoReady{false};
int MemSegment::deviceId_{-1};
int MemSegment::logicDeviceId_{-1};
uint32_t MemSegment::pid_{0};
uint32_t MemSegment::sdid_{0};
uint32_t MemSegment::serverId_{0};
uint32_t MemSegment::superPodId_{0};
AscendSocType MemSegment::socType_{AscendSocType::ASCEND_UNKNOWN};

MemSegmentPtr MemSegment::Create(const MemSegmentOptions &options, int entityId)
{
    if (options.rankId >= options.rankCnt) {
        SHM_LOG_ERROR("rank(" << options.rankId << ") but total " << options.rankCnt);
        return nullptr;
    }

    auto ret = MemSegment::InitDeviceInfo();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("MemSegment::InitDeviceInfo failed: " << ret);
        return nullptr;
    }

    MemSegmentPtr tmpSeg;
    switch (options.segType) {
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
        case HYBM_MST_HBM:
        case HYBM_MST_DRAM:
            tmpSeg = std::make_shared<MemSegmentDevice>(options, entityId);
            break;
#else
        case HYBM_MST_HBM:
            tmpSeg = std::make_shared<MemSegmentDevice>(options, entityId);
            break;
        case HYBM_MST_DRAM:
            SHM_LOG_ERROR("Not support HOST_SIDE malloc, please update CANN version");
            break;
#endif
        default:
            SHM_LOG_ERROR("Invalid memory seg type " << int(options.segType));
    }
    return tmpSeg;
}

bool MemSegment::CheckSdmaReaches(uint32_t rankId) const noexcept
{
    return false;
}

Result MemSegment::InitDeviceInfo()
{
    if (deviceInfoReady) {
        return ACLSHMEM_SUCCESS;
    }

    auto ret = DlAclApi::AclrtGetDevice(&deviceId_);
    if (ret != 0) {
        SHM_LOG_ERROR("get device id failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(deviceId_, &logicDeviceId_);
    if (ret != 0 || logicDeviceId_ < 0) {
        SHM_LOG_ERROR("Failed to get logic deviceId: " << deviceId_ << ", ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    ret = DlAclApi::RtDeviceGetBareTgid(&pid_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get bare tgid failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    int64_t value = 0;
    ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SDID, &value);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get sdid failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    sdid_ = static_cast<uint32_t>(value);
    ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SERVER_ID, &value);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get server id failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    serverId_ = static_cast<uint32_t>(value);
    SHM_LOG_DEBUG("local server=0x" << std::hex << serverId_);

    ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SUPER_POD_ID, &value);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get super pod id failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    superPodId_ = static_cast<uint32_t>(value);
    if (superPodId_ == invalidSuperPodId && serverId_ == invalidServerId) {
        auto networks = utils::NetworkGetIpAddresses();
        if (networks.empty()) {
            SHM_LOG_WARN("get local host ip address empty.");
        } else {
            serverId_ = networks[0];
        }
    }

    auto name = DlAclApi::AclrtGetSocName();
    if (name == nullptr) {
        SHM_LOG_ERROR("AclrtGetSocName() failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    std::string socName{name};
    if (socName.find("Ascend910B") != std::string::npos) {
        socType_ = AscendSocType::ASCEND_910B;
    } else if (socName.find("Ascend910_93") != std::string::npos) {
        socType_ = AscendSocType::ASCEND_910C;
    }

    SHM_LOG_DEBUG("local sdid=0x" << std::hex << sdid_ << ", local server=0x" << std::hex << serverId_
                                 << ", spid=" << superPodId_ << ", socName=" << socName);
    deviceInfoReady = true;
    return ACLSHMEM_SUCCESS;
}

bool MemSegment::CanLocalHostReaches(uint32_t superPodId, uint32_t serverId, uint32_t deviceId) noexcept
{
    if (superPodId != superPodId_ || serverId != serverId_) {
        return false;
    }
    return (socType_ != ASCEND_910B) || ((deviceId / ASC910B_CONN_RANKS) == (logicDeviceId_ / ASC910B_CONN_RANKS));
}

bool MemSegment::IsSdmaAccessible(uint32_t superPodId, uint32_t serverId, uint32_t deviceId) noexcept
{
    if (serverId == serverId_) {
        return (socType_ != ASCEND_910B) || ((deviceId / ASC910B_CONN_RANKS) == (logicDeviceId_ / ASC910B_CONN_RANKS));
    }

    if (superPodId == invalidSuperPodId || superPodId_ == invalidSuperPodId) {
        SHM_LOG_DEBUG("spid: " << superPodId << ", local: " << superPodId_ << " cannot reach.");
        return false;
    }

    return superPodId == superPodId_;
}
}