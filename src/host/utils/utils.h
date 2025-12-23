/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_UTILS_H
#define SHMEM_UTILS_H

#include <iostream>
#include <dlfcn.h>
#include "shmemi_logger.h"

class rtLibLoader {
private:
    void* rt_handle_;
    int (*rtGetLogicDevIdByUserDevIdFunc_)(const int32_t, int32_t *const);

    rtLibLoader() : rt_handle_(nullptr) {
        if (!loadLibrary()) {
            SHM_LOG_ERROR("Failed to initialize rtLibLoader: could not load liba.so or foo function");
        }
    }

    rtLibLoader(const rtLibLoader&) = delete;
    rtLibLoader& operator=(const rtLibLoader&) = delete;

    bool loadLibrary() {
        rt_handle_ = dlopen("libascendcl.so", RTLD_NOW);
        if (!rt_handle_) {
            SHM_LOG_ERROR("dlopen failed: " << dlerror());
            return false;
        }

        *((void**)&rtGetLogicDevIdByUserDevIdFunc_) = dlsym(rt_handle_, "rtGetLogicDevIdByUserDevId");
        if (!rtGetLogicDevIdByUserDevIdFunc_) {
            dlclose(rt_handle_);
            rt_handle_ = nullptr;
            SHM_LOG_ERROR("Unable to get info from " << "libascendcl.so" << ".");
            return ACLSHMEM_INVALID_VALUE;
        }
        return true;
    }
    
public:
    static rtLibLoader& getInstance() {
        static rtLibLoader instance;
        return instance;
    }

    void getLogicDevId(const int32_t userDeviceId, int32_t *const logicDeviceId) {
        if (rtGetLogicDevIdByUserDevIdFunc_) {
            rtGetLogicDevIdByUserDevIdFunc_(userDeviceId, logicDeviceId);
        } else {
            SHM_LOG_ERROR("rtGetLogicDevIdByUserDevIdFunc function is not available");
        }
    }

    bool isLoaded() const {
        return rt_handle_ != nullptr && rtGetLogicDevIdByUserDevIdFunc_ != nullptr;
    }

    ~rtLibLoader() {
        if (rt_handle_) {
            dlclose(rt_handle_);
        }
    }
};

#endif  // ACLSHMEM_UTILS_H