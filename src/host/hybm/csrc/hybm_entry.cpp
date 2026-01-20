/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>
#include <thread>
#include <mutex>
#include <fstream>
#include <limits.h>

#include "hybm_common_include.h"
#include "under_api/dl_api.h"
#include "under_api/dl_acl_api.h"
#include "under_api/dl_hal_api.h"
#include "devmm_svm_gva.h"
#include "hybm_cmd.h"
#include "hybm_driver.h"
#include "hybm.h"
#include "shmemi_file_util.h"

using namespace shm::hybm;

namespace {
const std::string DRIVER_VER_V4 = "V100R001C23SPC005B219";
const std::string DRIVER_VER_V3 = "V100R001C21B035";
const std::string DRIVER_VER_V2 = "V100R001C19SPC109B220";
const std::string DRIVER_VER_V1 = "V100R001C18B100";

static uint64_t g_baseAddr = 0ULL;
int64_t initialized = 0;
uint16_t initedDeviceId = 0;
int32_t initedLogicDeviceId = -1;
HybmGvaVersion checkVer = HYBM_GVA_UNKNOWN;
std::mutex initMutex;
}

int32_t HybmGetInitDeviceId()
{
    return static_cast<int32_t>(initedDeviceId);
}

bool HybmHasInited()
{
    return initialized > 0;
}

HybmGvaVersion HybmGetGvaVersion()
{
    return checkVer;
}

static bool DriverVersionCheck(const std::string &ver)
{
    auto libPath = std::getenv("LD_LIBRARY_PATH");
    if (libPath == nullptr) {
        BM_LOG_ERROR("check driver version failed, Environment LD_LIBRARY_PATH not set.");
        return false;
    }

    std::string readVer = CastDriverVersion(libPath);
    if (readVer.empty()) {
        BM_LOG_ERROR("check driver version failed, read version is empty.");
        return false;
    }

    int32_t baseVal = GetValueFromVersion(ver, "V");
    int32_t readVal = GetValueFromVersion(readVer, "V");
    if (baseVal == -1 || readVal == -1 || baseVal != readVal) {
        BM_LOG_INFO("check driver version failed, V Version not equal");
        return false;
    }

    baseVal = GetValueFromVersion(ver, "R");
    readVal = GetValueFromVersion(readVer, "R");
    if (baseVal == -1 || readVal == -1 || baseVal != readVal) {
        BM_LOG_INFO("check driver version failed, R Release not equal");
        return false;
    }

    baseVal = GetValueFromVersion(ver, "C");
    readVal = GetValueFromVersion(readVer, "C");
    if (baseVal == -1 || readVal == -1 || readVal < baseVal) {
        BM_LOG_INFO("check driver version failed, C Customer is too low");
        return false;
    }
    if (readVal > baseVal) {
        return true;
    }

    baseVal = GetValueFromVersion(ver, "SPC");
    readVal = GetValueFromVersion(readVer, "SPC");
    if (baseVal != -1) {
        if (readVal == -1 || readVal < baseVal) {
            BM_LOG_DEBUG("Driver version mismatch, base=" << baseVal << ", read=" << readVal);
            return false;
        }
        if (readVal > baseVal) {
            return true;
        }
    } else {
        if (readVal != -1) {
            return true;
        }
    }

    baseVal = GetValueFromVersion(ver, "B");
    readVal = GetValueFromVersion(readVer, "B");
    if (baseVal == -1 || readVal == -1 || readVal < baseVal) {
        BM_LOG_INFO("check driver version failed, B Build is too low:");
        return false;
    }
    return true;
}

int32_t HalGvaPrecheck(void)
{
    if (DriverVersionCheck(DRIVER_VER_V3)) {
        BM_LOG_INFO("Driver version V3 found");
        checkVer = HYBM_GVA_V3;
        return BM_OK;
    }
    if (DriverVersionCheck(DRIVER_VER_V2)) {
        BM_LOG_INFO("Driver version V2 found");
        checkVer = HYBM_GVA_V2;
        return BM_OK;
    }
    if (DriverVersionCheck(DRIVER_VER_V1)) {
        BM_LOG_INFO("Driver version V1 found");
        checkVer = HYBM_GVA_V1;
        return BM_OK;
    }
    BM_LOG_ERROR("Failed to determine driver version");
    return BM_ERROR;
}

static inline int hybm_load_library()
{
    char *path = std::getenv("ASCEND_HOME_PATH");
    BM_VALIDATE_RETURN(path != nullptr, "Environment ASCEND_HOME_PATH not set.", BM_ERROR);

    std::string libPath = std::string(path).append("/lib64");
    if (!shm::utils::FileUtil::Realpath(libPath) || !shm::utils::FileUtil::IsDir(libPath)) {
        BM_LOG_ERROR("Environment ASCEND_HOME_PATH check failed.");
        return BM_ERROR;
    }
    auto ret = DlApi::LoadLibrary(libPath);
    BM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "load library from path failed: " << ret);
    return 0;
}

HYBM_API int32_t hybm_init(uint16_t deviceId, uint64_t flags)
{
    std::unique_lock<std::mutex> lockGuard{initMutex};
    if (initialized > 0) {
        if (initedDeviceId != deviceId) {
            BM_LOG_ERROR("this deviceId(" << deviceId << ") is not equal to the deviceId(" <<
                initedDeviceId << ") of other module!");
            return BM_ERROR;
        }

        /*
         * hybm_init will be accessed multiple times when bm/shm/trans init
         * incremental loading is required here.
         */
        BM_LOG_ERROR_RETURN_IT_IF_NOT_OK(hybm_load_library(), "load library failed");

        initialized++;
        return 0;
    }

    BM_LOG_ERROR_RETURN_IT_IF_NOT_OK(HalGvaPrecheck(), "the current version of ascend driver does not support mf!");

    BM_LOG_ERROR_RETURN_IT_IF_NOT_OK(hybm_load_library(), "load library failed");

    auto ret = DlAclApi::RtGetLogicDevIdByUserDevId(deviceId, &initedLogicDeviceId);
    if (ret != 0 || initedLogicDeviceId < 0) {
        BM_LOG_ERROR("fail to get logic device id " << deviceId << ", ret=" << ret);
        return BM_ERROR;
    }
    BM_LOG_INFO("success to get logic device user id=" << deviceId << ", logic deviceId = " << initedLogicDeviceId);
    ret = DlAclApi::AclrtSetDevice(deviceId);
    if (ret != BM_OK) {
        DlApi::CleanupLibrary();
        BM_LOG_ERROR("set device id to be " << deviceId << " failed: " << ret);
        return BM_ERROR;
    }

    void *globalMemoryBase = nullptr;
    size_t allocSize = HYBM_DEVICE_INFO_SIZE;  // 申请meta空间
    drv::HybmInitialize(initedLogicDeviceId, DlHalApi::GetFd());
    ret = drv::HalGvaReserveMemory((uint64_t *)&globalMemoryBase, allocSize, initedLogicDeviceId, flags);
    if (ret != 0) {
        DlApi::CleanupLibrary();
        BM_LOG_ERROR("initialize mete memory with size: " << allocSize << ", flag: " << flags << " failed: " << ret);
        return BM_ERROR;
    }

    ret = drv::HalGvaAlloc(HYBM_DEVICE_META_ADDR, HYBM_DEVICE_INFO_SIZE, 0);
    if (ret != BM_OK) {
        DlApi::CleanupLibrary();
        int32_t hal_ret = drv::HalGvaUnreserveMemory((uint64_t)globalMemoryBase);
        BM_LOG_ERROR("HalGvaAlloc hybm meta memory failed: " << ret << ", un-reserve memory " << hal_ret);
        return BM_MALLOC_FAILED;
    }

    initedDeviceId = deviceId;
    BM_LOG_INFO("hybm_init end device id " << deviceId << ", logic device id " << initedLogicDeviceId);
    initialized = 1L;
    g_baseAddr = (uint64_t)globalMemoryBase;
    BM_LOG_INFO("hybm init successfully.");
    return 0;
}

HYBM_API void hybm_uninit(void)
{
    std::unique_lock<std::mutex> lockGuard{initMutex};
    if (initialized <= 0L) {
        BM_LOG_WARN("hybm not initialized.");
        return;
    }

    if (--initialized > 0L) {
        return;
    }

    drv::HalGvaFree(HYBM_DEVICE_META_ADDR, HYBM_DEVICE_INFO_SIZE);
    auto ret = drv::HalGvaUnreserveMemory(g_baseAddr);
    g_baseAddr = 0ULL;
    BM_LOG_INFO("uninitialize GVA memory return: " << ret);
    initialized = 0;
}
