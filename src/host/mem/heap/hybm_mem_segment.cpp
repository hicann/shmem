/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <sstream>

#include "dl_comm_def.h"
#include "dl_api.h"
#include "dl_acl_api.h"
#include "shmemi_net_util.h"
#include "host/shmem_host_def.h"
#include "hybm_device_mem_segment.h"
#include "hybm_vmm_based_segment.h"

namespace shm {
bool MemSegment::deviceInfoReady{false};
int MemSegment::deviceId_{-1};
int MemSegment::logicDeviceId_{-1};
int MemSegment::devicePhyId_{-1};
uint32_t MemSegment::pid_{0};
uint32_t MemSegment::sdid_{0};
uint32_t MemSegment::serverId_{0};
bool MemSegment::serverIdSubstituted_{false};
uint32_t MemSegment::superPodId_{0};
AscendSocType MemSegment::socType_{AscendSocType::ASCEND_UNKNOWN};
std::string MemSegment::sysBoolId_{};
uint32_t MemSegment::bootIdHead_{0};

MemSegmentPtr MemSegment::Create(const MemSegmentOptions& options, int entityId)
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
            if (options.userBufferHeapInput != nullptr) {
                tmpSeg = std::make_shared<AclMemSegmentDevice>(options, entityId);
                break;
            }
#ifdef USE_ACLRT_MEM_FABRIC_HANDLE
            tmpSeg = std::make_shared<AclMemSegmentDevice>(options, entityId);
#else
            if (socType_ == AscendSocType::ASCEND_950 || (HybmGetGvaVersion() == HYBM_GVA_V4)) {
                tmpSeg = std::make_shared<HybmVmmBasedSegment>(options, entityId);
            } else {
                tmpSeg = std::make_shared<MemSegmentDevice>(options, entityId);
            }
#endif
            break;
        case HYBM_MST_DRAM:
#ifdef USE_ACLRT_MEM_FABRIC_HANDLE
            tmpSeg = std::make_shared<AclMemSegmentDevice>(options, entityId);
#else
            SHM_LOG_ERROR("Not support HOST_SIDE malloc now.");
#endif
            break;
#else
        case HYBM_MST_HBM:
            if (socType_ == AscendSocType::ASCEND_950 || (HybmGetGvaVersion() == HYBM_GVA_V4)) {
                tmpSeg = std::make_shared<HybmVmmBasedSegment>(options, entityId);
            } else {
                tmpSeg = std::make_shared<MemSegmentDevice>(options, entityId);
            }
            break;
        case HYBM_MST_DRAM:
            SHM_LOG_ERROR("Not support HOST_SIDE malloc now.");
            break;
#endif
        default:
            SHM_LOG_ERROR("Invalid memory seg type " << int(options.segType));
    }
    return tmpSeg;
}

Result MemSegment::GetDeviceInfo(uint32_t& sdId, uint32_t& serverId, uint32_t& superPodId)
{
    auto ret = MemSegment::InitDeviceInfo();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("MemSegment::InitDeviceInfo failed: " << ret);
        return ret;
    }

    sdId = sdid_;
    serverId = serverId_;
    superPodId = superPodId_;
    return ACLSHMEM_SUCCESS;
}

bool MemSegment::CheckSdmaReaches(uint32_t rankId) const noexcept { return false; }

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

    ret = DlAclApi::AclrtGetPhyDevIdByUserDevId(deviceId_, &devicePhyId_);
    if (ret != 0 || devicePhyId_ < 0) {
        SHM_LOG_ERROR(
            "Failed to get phy deviceId: user=" << deviceId_ << ", logic=" << logicDeviceId_ << ", ret=" << ret);
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
    socType_ = DlApi::GetAscendSocType();

    FillSysBootIdInfo();
    superPodId_ = static_cast<uint32_t>(value);
    const uint32_t rawServerId = serverId_;
    if (superPodId_ == invalidSuperPodId && serverId_ == GetInvalidServerIdBySocType(socType_)) {
        if (bootIdHead_ != 0) {
            serverId_ = bootIdHead_;
        } else {
            auto networks = utils::NetworkGetIpAddresses();
            if (networks.empty()) {
                SHM_LOG_ERROR("get local host ip address empty.");
                return ACLSHMEM_INNER_ERROR;
            }
            serverId_ = networks[0];
        }
        serverIdSubstituted_ = true;
        SHM_LOG_WARN(
            "serverId read the invalid sentinel 0x"
            << std::hex << rawServerId << ", substituted with 0x" << serverId_ << std::dec
            << (bootIdHead_ != 0 ? " (boot-id head)" : " (local IP)")
            << ". This id identifies the OS instance rather than the physical server, so only PEs "
               "sharing it are judged same-server.");
    }
    if (superPodId_ == invalidSuperPodId) {
        SHM_LOG_WARN(
            "superPodId read invalid (0x"
            << std::hex << invalidSuperPodId << std::dec << ") on logicDevId=" << logicDeviceId_
            << ". Every peer on a different server will be rejected for SDMA/MTE no matter what the real "
               "fabric topology is, and its heap slot will stay unmapped. This is expected only for a "
               "deployment without a super pod; otherwise the super pod is not provisioned on this node.");
    }
    SHM_LOG_DEBUG(
        "local topology identity: sdid=0x"
        << std::hex << sdid_ << ", server=0x" << serverId_ << std::dec << (serverIdSubstituted_ ? "(substituted)" : "")
        << ", spid=" << superPodId_ << (superPodId_ == invalidSuperPodId ? "(INVALID)" : "")
        << ", logicDevId=" << logicDeviceId_ << ", socType=" << static_cast<int>(socType_));
    deviceInfoReady = true;
    return ACLSHMEM_SUCCESS;
}

void MemSegment::FillSysBootIdInfo() noexcept
{
    std::string bootIdPath("/proc/sys/kernel/random/boot_id");
    std::ifstream input(bootIdPath);
    input >> sysBoolId_;

    std::stringstream ss(sysBoolId_);
    ss >> std::hex >> bootIdHead_;
    SHM_LOG_DEBUG("os-boot-id: " << sysBoolId_ << ", head u32: " << std::hex << bootIdHead_);
}

uint32_t MemSegment::GetInvalidServerIdBySocType(AscendSocType socType) noexcept
{
    return (socType == AscendSocType::ASCEND_950) ? invalidServerIdAscend950 : invalidServerId;
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

bool MemSegmentDevice::LogRemoteUnreachable(const HbmExportInfo& rmi) noexcept
{
    const bool localSpidUnset = (superPodId_ == invalidSuperPodId);
    const bool remoteSpidUnset = (rmi.superPodId == invalidSuperPodId);
    const bool remoteSidInvalid = (rmi.serverId == GetInvalidServerIdBySocType(socType_));
    const bool remoteDevIdBad = (rmi.logicDeviceId < 0);
    // serverIdSubstituted_ is deliberately not a term here: substitution only happens when superPodId_ is
    // already invalid, so localSpidUnset always covers it. The flag stays as a log annotation only.
    const bool suspicious = localSpidUnset || remoteSpidUnset || remoteSidInvalid || remoteDevIdBad;

    std::ostringstream os;
    os << "SDMA/MTE cannot reach rank " << rmi.rankId << ", its heap slot stays unmapped."
       << " | local: spid=" << superPodId_ << (localSpidUnset ? "(INVALID)" : "") << " server=0x" << std::hex
       << serverId_ << std::dec << (serverIdSubstituted_ ? "(substituted)" : "") << " logicDevId=" << logicDeviceId_
       << " socType=" << static_cast<int>(socType_) << " | remote: spid=" << rmi.superPodId
       << (remoteSpidUnset ? "(INVALID)" : "") << " server=0x" << std::hex << rmi.serverId << std::dec
       << (remoteSidInvalid ? "(INVALID)" : "") << " logicDevId=" << rmi.logicDeviceId
       << (remoteDevIdBad ? "(NEGATIVE)" : "");

    if (suspicious) {
        os << " | this identity looks incomplete";
        SHM_LOG_WARN(os.str());
        return true;
    }
    // both identities are well-formed, so this is a genuine cross-super-pod peer
    SHM_LOG_INFO(os.str());
    return false;
}

void MemSegmentDevice::LogUnreachableSummary(uint32_t unreachableCnt, bool anySuspicious) noexcept
{
    if (unreachableCnt == 0) {
        return;
    }

    std::ostringstream os;
    os << unreachableCnt
       << " peer(s) left unmapped. A peer is mappable only when (1) its serverId equals "
          "the local one, and on 910B its logicDevId/"
       << ASC910B_CONN_RANKS
       << " also matches, or (2) its serverId differs while both superPodIds are valid and equal.";
    if (!anySuspicious) {
        SHM_LOG_INFO(os.str());
        return;
    }
    os << " An invalid superPodId means no super pod is provisioned on that node; if these PEs are meant "
          "to share one, check the super-pod configuration rather than this heap.";
    SHM_LOG_WARN(os.str());
}

Result MemSegment::EnableRemotePeerAccess(int32_t remotePhyId, int32_t remoteUserId) noexcept
{
    if (remotePhyId < 0) {
        SHM_LOG_ERROR("invalid remote phy id: " << remotePhyId);
        return ACLSHMEM_INVALID_PARAM;
    }
    if (remotePhyId == devicePhyId_) {
        return ACLSHMEM_SUCCESS;
    }

    Result ret = DlAclApi::RtEnableP2P(static_cast<uint32_t>(deviceId_), static_cast<uint32_t>(remotePhyId), 0);
    if (ret == ACLSHMEM_UNDER_API_UNLOAD) {
        if (remoteUserId < 0) {
            SHM_LOG_ERROR("invalid remote user id: " << remoteUserId);
            return ACLSHMEM_INVALID_PARAM;
        }
        ret = DlAclApi::AclrtDeviceEnablePeerAccess(remoteUserId, 0);
        if (ret != 0) {
            SHM_LOG_ERROR(
                "enable device access failed:" << ret << " local_user:" << deviceId_ << " local_phy:" << devicePhyId_
                                               << " remote_user:" << remoteUserId << " remote_phy:" << remotePhyId);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        SHM_LOG_DEBUG(
            "enable device access success (aclrt fallback) local_user:" << deviceId_
                                                                        << " remote_user:" << remoteUserId);
        return ACLSHMEM_SUCCESS;
    }
    if (ret != 0) {
        SHM_LOG_ERROR(
            "enable device access failed:" << ret << " local_user:" << deviceId_ << " local_phy:" << devicePhyId_
                                           << " remote_phy:" << remotePhyId);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    SHM_LOG_DEBUG(
        "enable device access success local_user:" << deviceId_ << " local_phy:" << devicePhyId_
                                                   << " remote_phy:" << remotePhyId);
    return ACLSHMEM_SUCCESS;
}
} // namespace shm
