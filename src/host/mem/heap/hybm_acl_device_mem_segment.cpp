/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE

#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"

#include "dl_comm_def.h"
#include "mem_entity_def.h"
#include "devmm_svm_gva.h"
#include "shmemi_logger.h"
#include "shmemi_net_util.h"
#include "hybm_ex_info_transfer.h"
#include "hybm_device_mem_segment.h"

namespace shm {
std::string MemSegmentDevice::sysBoolId_;
uint32_t MemSegmentDevice::bootIdHead_{0};

Result MemSegmentDevice::ValidateOptions() noexcept
{
    if (options_.size == 0 || options_.devId < 0 ||
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
    void *base = nullptr;
    for (uint32_t i = 0; i < options_.rankCnt; i++) {
        auto ret = aclrtReserveMemAddress(&base, options_.size, 0, nullptr, 1);
        if (ret != 0 || base == 0) {
            SHM_LOG_ERROR("prepare virtual memory size(" << totalVirtualSize_ << ") failed. ret: " << ret);
            return ACLSHMEM_MALLOC_FAILED;
        }
        SHM_LOG_INFO("success to reserve memory space for logic deviceid " << logicDeviceId_ << ", vaddr: " << (void *)base << " size: " << options_.size << ", rankId: " << i);
        totalVirtualSize_ += options_.size;
        reservedVirtualAddresses_.emplace_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base)));
    }

    globalVirtualAddress_ = reinterpret_cast<uint8_t *>(reservedVirtualAddresses_[0]);
    allocatedSize_ = 0UL;
    sliceCount_ = 0;
    *address = reinterpret_cast<void *>(reservedVirtualAddresses_[0]);
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::SetMemAccess() noexcept
{
    if (options_.segType == HYBM_MST_DRAM) {
        aclrtMemLocation loc;
        loc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        loc.id = options_.rankId;
        aclrtMemAccessDesc des;
        des.location = loc;
        des.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
        ACLSHMEM_CHECK_RET(aclrtMemSetAccess(reinterpret_cast<void *>(localVirtualBase), options_.size, &des, 1),
            "aclrtMemSetAccess failed.", ACLSHMEM_SMEM_ERROR);
    }
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::UnReserveMemorySpace() noexcept
{
    SHM_LOG_INFO("un-reserve memory space.");
    FreeMemory();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::AllocLocalMemory(uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    if ((size % DEVICE_LARGE_PAGE_SIZE) != 0UL || size + allocatedSize_ > options_.size) {
        SHM_LOG_ERROR("invalid allocate memory size : " << size << ", now used " << allocatedSize_ << " of "
                                                       << options_.size);
        return ACLSHMEM_INVALID_PARAM;
    }
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = (options_.segType == HYBM_MST_DRAM) ? 0 : deviceId_;
    prop.location.type = (options_.segType == HYBM_MST_DRAM) ? ACL_MEM_LOCATION_TYPE_HOST : ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;
    auto ret = aclrtMallocPhysical(&local_handle_, size, &prop, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(options_.rankId << " aclrtMallocPhysical memory failed: " << ret << " segType: " << (int)options_.segType);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    SHM_LOG_DEBUG(options_.rankId << " aclrtMallocPhysical memory success size: " << size << " vaddr: " << reinterpret_cast<void *>(localVirtualBase) << " segType: " << (int)options_.segType);
    ACLSHMEM_CHECK_RET(aclrtMapMem(reinterpret_cast<void *>(localVirtualBase), size, 0, local_handle_, 0));

    if (SetMemAccess() != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_SMEM_ERROR;
    }

    slice = std::make_shared<MemSlice>(sliceCount_++, MEM_TYPE_DEVICE_HBM, MEM_PT_TYPE_SVM,
                                       reinterpret_cast<uint64_t>(localVirtualBase), size);
    slices_.emplace(slice->index_, slice);
    SHM_LOG_DEBUG("allocate slice(idx:" << slice->index_ << ", size:" << slice->size_ << ").");
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::RegisterMemory(const void *addr, uint64_t size, std::shared_ptr<MemSlice> &slice) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice NOT SUPPORT RegisterMemory");
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::ReleaseSliceMemory(const std::shared_ptr<MemSlice> &slice) noexcept
{
    SHM_VALIDATE_RETURN(slice != nullptr, "input slice is nullptr", ACLSHMEM_INVALID_PARAM);

    auto pos = slices_.find(slice->index_);
    if (pos == slices_.end()) {
        SHM_LOG_ERROR("input slice(idx:" << slice->index_ << ") not exist.");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (pos->second.slice != slice) {
        SHM_LOG_ERROR("input slice(magic:" << std::hex << slice->magic_ << ") not match.");
        return ACLSHMEM_INVALID_PARAM;
    }
    auto res = aclrtUnmapMem(reinterpret_cast<void *>(slice->vAddress_));
    SHM_LOG_INFO("unmap slice(idx:" << slice->index_ << "), size: " << slice->size_ << " return:" << res);

    slices_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Export(std::string &exInfo) noexcept
{
    SHM_LOG_INFO("MemSegmentDevice not supported export device info.");
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Export(const std::shared_ptr<MemSlice> &slice, std::string &exInfo) noexcept
{
    if (slice == nullptr) {
        SHM_LOG_ERROR("input slice is nullptr");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip export");
        return ACLSHMEM_SUCCESS;
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
    if (exp != exportMap_.end()) {
        exInfo = exp->second;
        return ACLSHMEM_SUCCESS;
    }

    HbmExportInfo info{};
    auto handle_type = (options_.segType == HYBM_MST_DRAM) ? ACL_MEM_SHARE_HANDLE_TYPE_FABRIC : ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
    auto ret = aclrtMemExportToShareableHandleV2(local_handle_, 1, handle_type, &info.shareHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtMemExportToShareableHandleV2 failed, ret: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(SetDeviceInfo(options_.devId), "get device info failed.");
    auto localVirtualBase = reservedVirtualAddresses_[options_.rankId];
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
    info.memSegType = options_.segType;
    info.exchangeType = HYBM_INFO_EXG_IN_NODE;
    info.magic = (options_.segType == HYBM_MST_DRAM) ? DRAM_SLICE_EXPORT_INFO_MAGIC : HBM_SLICE_EXPORT_INFO_MAGIC;
    ret = LiteralExInfoTranslater<HbmExportInfo>{}.Serialize(info, exInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("export info failed: " << ret);
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
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip import");
        return ACLSHMEM_SUCCESS;
    }
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
    std::copy(desInfos.begin(), desInfos.end(), std::back_inserter(imports_));
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Mmap() noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip map");
        return ACLSHMEM_SUCCESS;
    }
    if (imports_.empty()) {
        return ACLSHMEM_SUCCESS;
    }
    for (auto &im : imports_) {
        if (im.rankId == options_.rankId) {
            continue;
        }

        size_t reserveAlignedSize = ALIGN_UP(options_.size, DEVMM_HEAP_SIZE);
        auto remoteAddress = reservedVirtualAddresses_[im.rankId];;
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
        aclrtDrvMemHandle imported_handle;
        auto handle_type = (options_.segType == HYBM_MST_DRAM) ? ACL_MEM_SHARE_HANDLE_TYPE_FABRIC : ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT;
        ACLSHMEM_CHECK_RET(aclrtMemImportFromShareableHandleV2(&im.shareHandle, handle_type, 0, &imported_handle),
            "aclrtMemImportFromShareableHandleV2 failed", ACLSHMEM_SMEM_ERROR);
        auto ret = aclrtMapMem(reinterpret_cast<void *>(remoteAddress), im.size, 0, imported_handle, 0);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("aclrtMapMem memory failed:" << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        mappedMem_.insert((uint64_t)remoteAddress);
    }
    imports_.clear();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::Unmap() noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip unmap");
        return ACLSHMEM_SUCCESS;
    }
    for (auto va : mappedMem_) {
        int32_t ret = aclrtUnmapMem(reinterpret_cast<void *>(va));
        if (ret != 0) {
            SHM_LOG_ERROR("aclrtUnmapMem memory failed:" << ret);
        }
    }
    mappedMem_.clear();
    return ACLSHMEM_SUCCESS;
}

Result MemSegmentDevice::RemoveImported(const std::vector<uint32_t> &ranks) noexcept
{
    if (!options_.shared) {
        SHM_LOG_INFO("no need to share, skip remove");
        return ACLSHMEM_SUCCESS;
    }
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
            int32_t ret = aclrtUnmapMem(reinterpret_cast<void *>(*it));
            if (ret != 0) {
                SHM_LOG_ERROR("aclrtUnmapMem failed " << ret);
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
    aclrtFreePhysical(local_handle_);
    allocatedSize_ = 0;
    sliceCount_ = 0;
    if (globalVirtualAddress_ != nullptr) {
        for (auto reserved : reservedVirtualAddresses_) {
            aclrtReleaseMemAddress(reinterpret_cast<void *>(reserved));
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

    int32_t tgid = 0;
    auto ret = aclrtDeviceGetBareTgid(&tgid);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get bare tgid failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    deviceId_ = deviceId;
    pid_ = static_cast<uint32_t>(tgid);
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
    auto ret = rtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SDID, &value);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get sdid failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    sdid_ = static_cast<uint32_t>(value);

    ret = rtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SERVER_ID, &value);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("get server id failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    serverId_ = static_cast<uint32_t>(value);
    SHM_LOG_DEBUG("local server=0x" << std::hex << serverId_);

    ret = rtGetDeviceInfo(deviceId_, 0, INFO_TYPE_SUPER_POD_ID, &value);
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

    auto name = aclrtGetSocName();
    if (name == nullptr) {
        SHM_LOG_ERROR("aclrtGetSocName() failed.");
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
    return true;
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