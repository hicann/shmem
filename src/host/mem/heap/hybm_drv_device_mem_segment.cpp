/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "dl_comm_def.h"
#include "dl_api.h"
#include "dl_hal_api.h"
#include "dl_acl_api.h"
#include "shmemi_logger.h"
#include "devmm_svm_gva.h"
#include "mem_entity_def.h"
#include "shmemi_net_util.h"
#include "hybm_ex_info_transfer.h"
#include "hybm_device_mem_segment.h"

#ifndef HAS_ACLRT_MEM_FABRIC_HANDLE

namespace shm {
std::string MemSegmentDevice::sysBoolId_;
uint32_t MemSegmentDevice::bootIdHead_{0};

Result MemSegmentDevice::ValidateOptions() noexcept
{
    if (options_.segType != HYBM_MST_HBM || options_.size == 0 || options_.devId < 0 ||
        (options_.size % DEVICE_LARGE_PAGE_SIZE) != 0) {
        return ACLSHMEM_INVALID_PARAM;
    }

    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::ReserveMemorySpace(void **address) noexcept
{
    if (globalVirtualAddress_ != nullptr) {
        SHM_LOG_ERROR("already prepare virtual memory.");
        return ACLSHMEM_INNER_ERROR;
    }

    uint64_t base = 0;
    for (uint32_t i = 0; i < options_.rankCnt; i++) {
        auto ret = drv::HalGvaReserveMemory(&base, options_.size, logicDeviceId_, 0ULL);
        if (ret != 0 || base == 0) {
            SHM_LOG_ERROR("prepare virtual memory size to (" << totalVirtualSize_ << ") failed. ret: " << ret);
            return ACLSHMEM_MALLOC_FAILED;
        }
        SHM_LOG_INFO("success to reserve memory space for logic deviceid " << logicDeviceId_ << ", vaddr: " << (void *)base << ", size: " << options_.size << ", rankId: " << i);
        size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        totalVirtualSize_ += reserveAlignedSize;
        reservedVirtualAddresses_.emplace_back(base);
    }

    globalVirtualAddress_ = reinterpret_cast<uint8_t *>(base);
    allocatedSize_ = 0UL;
    sliceCount_ = 0;
    *address = reinterpret_cast<void *>(base);
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::UnReserveMemorySpace() noexcept
{
    SHM_LOG_INFO("un-reserve memory space.");
    FreeMemory();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::SetMemAccess() noexcept
{
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0UL || size + allocatedSize_ > options_.size) {
        SHM_LOG_ERROR("invalid allocate memory size : " << size << ", now used " << allocatedSize_ << " of "
                                                       << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }

    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    auto localVirtualBase = globalVirtualAddress_ + reserveAlignedSize * options_.rankId;
    auto ret = drv::HalGvaAlloc((uint64_t)(localVirtualBase + allocatedSize_), size, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("HalGvaAlloc memory failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    auto sliceAddr = localVirtualBase + allocatedSize_;
    allocatedSize_ += size;
    slice = std::make_shared<MemSlice>(sliceCount_++, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_SVM,
                                       static_cast<uint64_t>(reinterpret_cast<std::ptrdiff_t>(sliceAddr)), size);
    slices_.emplace(slice->index_, slice);
    SHM_LOG_DEBUG("allocate slice(idx:" << slice->index_ << ", size:" << slice->size_ << ").");

    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::RegisterMemory(const void *addr, uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    SHM_LOG_ERROR("MemSegmentDevice NOT SUPPORT RegisterMemory");
    return ACLSHMEM_NOT_SUPPORTED;
}

Result MemSegmentDevice::ReleaseSliceMemory(const std::shared_ptr<MemSlice> &slice) noexcept
{
    if (slice == nullptr) {
        SHM_LOG_ERROR("input slice is nullptr");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto res = drv::HalGvaFree(slice->vAddress_, slice->size_);
    SHM_LOG_INFO("free slice(idx:" << slice->index_ << "), size: " << slice->size_ << " return:" << res);

    slices_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Export(std::string &exInfo) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice not supported export device info.");
    return ACLSHMEM_SUCCESS;
}

// export不可重入
Result MemSegmentDevice::Export(const std::shared_ptr<MemSlice> &slice, std::string &exInfo) noexcept
{
    if (slice == nullptr) {
        SHM_LOG_ERROR("input slice is nullptr");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto exp = exportMap_.find(slice->index_);
    if (exp != exportMap_.end()) {  // RtIpcSetMemoryName不支持重复调用
        exInfo = exp->second;
        return ACLSHMEM_SUCCESS;
    }

    HbmExportInfo info{};
    auto ret = DlAclApi::RtIpcSetMemoryName((void *)(ptrdiff_t)slice->vAddress_, slice->size_, info.shmName,
                                            sizeof(info.shmName));
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "set memory name failed: " << ret);

    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(SetDeviceInfo(options_.devId), "get device info failed.");
    size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
    auto localVirtualBase = globalVirtualAddress_ + reserveAlignedSize * options_.rankId;
    info.mappingOffset = slice->vAddress_ - (uint64_t)(ptrdiff_t)localVirtualBase;
    info.sliceIndex = static_cast<uint32_t>(slice->index_);
    info.deviceId = options_.devId;
    info.logicDeviceId = logicDeviceId_;
    info.pid = pid_;
    info.rankId = options_.rankId;
    info.size = slice->size_;
    info.entityId = entityId_;
    info.sdid = sdid_;
    info.serverId = serverId_;
    info.superPodId = superPodId_;
    info.pageTblType = MEM_PT_TYPE_SVM;
    info.memSegType = HYBM_MST_HBM;
    info.exchangeType = HYBM_INFO_EXG_IN_NODE;
    ret = LiteralExInfoTranslater<HbmExportInfo>{}.Serialize(info, exInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        auto acl_ret = DlAclApi::RtIpcDestroyMemoryName(info.shmName);
        SHM_LOG_ERROR("export info failed: " << ret << ", ipc destroy memory name " << acl_ret);
        return ACLSHMEM_INNER_ERROR;
    }

    exportMap_[slice->index_] = exInfo;
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::GetExportSliceSize(size_t &size) noexcept
{
    size = sizeof(HbmExportInfo);
    return ACLSHMEM_SUCCESS;
}

// import可重入
Result MemSegmentDevice::Import(const std::vector<std::string> &allExInfo, void *addresses[]) noexcept
{
    std::map<uint16_t, HbmExportInfo> importMap;
    LiteralExInfoTranslater<HbmExportInfo> translator;
    std::vector<HbmExportInfo> desInfos{allExInfo.size()};
    for (auto i = 0U; i < allExInfo.size(); i++) {
        auto ret = translator.Deserialize(allExInfo[i], desInfos[i]);
        if (ret != 0) {
            SHM_LOG_ERROR("deserialize imported info(" << i << ") failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
        importMap.emplace(desInfos[i].rankId, desInfos[i]);
    }
    importMap_ = std::move(importMap);

    uint32_t localIdx = UINT32_MAX;
    for (auto i = 0U; i < desInfos.size(); i++) {
        if (desInfos[i].magic != EXPORT_INFO_MAGIC) {
            SHM_LOG_ERROR("import info(" << i << ") magic(" << desInfos[i].magic << ") invalid.");
            return ACLSHMEM_INVALID_PARAM;
        }

        if (desInfos[i].size != desInfos[0].size) {
            SHM_LOG_ERROR("local size diffs, pe0=" << desInfos[0].size << ", pe" << i << "=" << desInfos[i].size);
            return ACLSHMEM_INNER_ERROR;
        }

        if (desInfos[i].rankId == options_.rankId) {
            localIdx = i;
        }
    }
    SHM_ASSERT_RETURN(localIdx < desInfos.size(), ACLSHMEM_INVALID_PARAM);

    for (auto i = 0U; i < desInfos.size(); i++) {
        if (desInfos[i].rankId == options_.rankId) {
            continue;
        }

        if (CanLocalHostReaches(desInfos[i].superPodId, desInfos[i].serverId, desInfos[i].logicDeviceId)) {
            auto ret = DlAclApi::AclrtDeviceEnablePeerAccess(desInfos[i].deviceId, 0);
            if (ret != 0) {
                SHM_LOG_ERROR("enable device access failed:" << ret << " local_device:" << deviceId_
                                                            << " remote_device:" << desInfos[i].deviceId
                                                            << " logic_device:" << logicDeviceId_
                                                            << " remote_logic:" << desInfos[i].logicDeviceId);
                return ACLSHMEM_DL_FUNC_FAILED;
            }
        }

        if (!IsSdmaAccessible(desInfos[i].superPodId, desInfos[i].serverId, desInfos[i].logicDeviceId)) {
            continue;
        }

        auto ret = DlAclApi::RtSetIpcMemorySuperPodPid(desInfos[localIdx].shmName, desInfos[i].sdid,
                                                       &desInfos[i].pid, 1);
        if (ret != 0) {
            SHM_LOG_ERROR("enable white list for rank(" << desInfos[i].rankId << ") failed: " << ret
                                                       << ", local rank = " << options_.rankId
                                                       << ", shmName=" << desInfos[localIdx].shmName);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
    }

    std::copy(desInfos.begin(), desInfos.end(), std::back_inserter(imports_));
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Mmap() noexcept
{
    if (imports_.empty()) {
        return ACLSHMEM_SUCCESS;
    }

    for (auto &im : imports_) {
        if (im.rankId == options_.rankId) {
            continue;
        }

        size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        auto remoteAddress = globalVirtualAddress_ + reserveAlignedSize * im.rankId + im.mappingOffset;
        if (mappedMem_.find((uint64_t)remoteAddress) != mappedMem_.end()) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") has maped: " << (void *)remoteAddress);
            continue;
        }

        if (!CanMapRemote(im)) {
            SHM_LOG_INFO("remote slice on rank(" << im.rankId << ") SDMA cannot reaches.");
            continue;
        }

        SHM_LOG_DEBUG("remote slice on rank(" << im.rankId << ") should map to: " << (void *)remoteAddress
                                             << ", size = " << im.size);
        auto ret = drv::HalGvaOpen((uint64_t)remoteAddress, im.shmName, im.size, 0);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("HalGvaOpen memory failed:" << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        mappedMem_.insert((uint64_t)remoteAddress);
    }
    imports_.clear();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Unmap() noexcept
{
    for (auto va : mappedMem_) {
        int32_t ret = drv::HalGvaClose(va, 0);
        if (ret != 0) {
            SHM_LOG_ERROR("HalGvaClose memory failed:" << ret);
        }
    }
    mappedMem_.clear();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::RemoveImported(const std::vector<uint32_t> &ranks) noexcept
{
    for (auto &rank : ranks) {
        if (rank >= options_.rankCnt) {
            SHM_LOG_ERROR("input rank is invalid! rank:" << rank << " rankSize:" << options_.rankCnt);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    for (auto &rank : ranks) {
        size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        uint64_t addr = reinterpret_cast<uint64_t>(globalVirtualAddress_) + reserveAlignedSize * rank;
        auto it = mappedMem_.lower_bound(addr);
        auto st = it;
        while (it != mappedMem_.end() && (*it) < addr + reserveAlignedSize) {
            int32_t ret = drv::HalGvaClose((*it), 0);
            if (ret != 0) {
                SHM_LOG_ERROR("HalGvaClose failed " << ret);
            }
            it++;
        }

        if (st != it) {
            mappedMem_.erase(st, it);
        }
    }
    return 0;
}

std::shared_ptr<MemSlice> MemSegmentDevice::GetMemSlice(hybm_mem_slice_t slice) const noexcept
{
    auto index = MemSlice::GetIndexFrom(slice);
    auto pos = slices_.find(index);
    if (pos == slices_.end()) {
        return nullptr;
    }

    auto target = pos->second.slice;
    if (!target->ValidateId(slice)) {
        return nullptr;
    }

    return target;
}

bool MemSegmentDevice::MemoryInRange(const void *begin, uint64_t size) const noexcept
{
    if (begin < globalVirtualAddress_) {
        return false;
    }

    if (static_cast<const uint8_t *>(begin) + size >= globalVirtualAddress_ + totalVirtualSize_) {
        return false;
    }

    return true;
}

void MemSegmentDevice::FreeMemory() noexcept
{
    while (!slices_.empty()) {
        auto slice = slices_.begin()->second.slice;
        ReleaseSliceMemory(slice);
    }

    allocatedSize_ = 0;
    sliceCount_ = 0;
    if (globalVirtualAddress_ != nullptr) {
        for (auto reserved : reservedVirtualAddresses_) {
            auto ret = drv::HalGvaUnreserveMemory((uint64_t)reserved);
            if (ret != 0) {
                SHM_LOG_ERROR("HalGvaUnreserveMemory failed: " << ret);
            }
        }

        reservedVirtualAddresses_.clear();
        totalVirtualSize_ = 0;
        globalVirtualAddress_ = nullptr;
    }
}

Result MemSegmentDevice::GetDeviceInfo() noexcept
{
    if (options_.devId < 0) {
        return ACLSHMEM_INVALID_PARAM;
    }

    if (InitDeviceInfo() != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

int MemSegmentDevice::SetDeviceInfo(int deviceId) noexcept
{
    if (deviceId < 0) {
        return ACLSHMEM_INVALID_PARAM;
    }

    if (deviceId_ >= 0) {
        if (deviceId == deviceId_) {
            return 0;
        }

        return ACLSHMEM_INVALID_PARAM;
    }

    uint32_t tgid = 0;
    auto ret = DlAclApi::RtDeviceGetBareTgid(&tgid);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get bare tgid failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    deviceId_ = deviceId;
    pid_ = tgid;
    FillSysBootIdInfo();
    ret = FillDeviceSuperPodInfo();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("FillDeviceSuperPodInfo() failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

int MemSegmentDevice::FillDeviceSuperPodInfo() noexcept
{
    int64_t value = 0;

    auto ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SDID, &value);
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
        << ", spid=" << superPodId_ << ", socType=" << socType_ << ", devid=" << deviceId_);

    return ACLSHMEM_SUCCESS;
}

void MemSegmentDevice::FillSysBootIdInfo() noexcept
{
    std::string bootIdPath("/proc/sys/kernel/random/boot_id");
    std::ifstream input(bootIdPath);
    input >> sysBoolId_;

    std::stringstream ss(sysBoolId_);
    ss >> std::hex >> bootIdHead_;
    SHM_LOG_DEBUG("os-boot-id: " << sysBoolId_ << ", head u32: " << std::hex << bootIdHead_);
}

bool MemSegmentDevice::CanMapRemote(const HbmExportInfo &rmi) noexcept
{
    return IsSdmaAccessible(rmi.superPodId, rmi.serverId, rmi.logicDeviceId);
}

void MemSegmentDevice::GetDeviceInfo(uint32_t &sdId, uint32_t &serverId, uint32_t &superPodId) noexcept
{
    sdId = sdid_;
    serverId = serverId_;
    superPodId = superPodId_;
}

bool MemSegmentDevice::GetRankIdByAddr(const void *addr, uint64_t size, uint32_t &rankId) const noexcept
{
    auto end = static_cast<const uint8_t *>(addr) + size;
    if (addr >= globalVirtualAddress_ && end < globalVirtualAddress_ + totalVirtualSize_) {
        auto offset = static_cast<const uint8_t *>(addr) - static_cast<const uint8_t *>(globalVirtualAddress_);
        auto reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        auto alignOffset = offset % reserveAlignedSize;
        if (alignOffset < options_.size) {
            rankId = offset / reserveAlignedSize;
            return true;
        }
    }

    SHM_LOG_ERROR("input address, size: " << size << " cannot matches rankId.");
    rankId = std::numeric_limits<uint32_t>::max();
    return false;
}

bool MemSegmentDevice::CheckSdmaReaches(uint32_t remoteRankId) const noexcept
{
    auto pos = importMap_.find(static_cast<uint16_t>(remoteRankId));
    if (pos == importMap_.end()) {
        return false;
    }

    return IsSdmaAccessible(pos->second.superPodId, pos->second.serverId, pos->second.logicDeviceId);
}
}  // namespace shm

#endif