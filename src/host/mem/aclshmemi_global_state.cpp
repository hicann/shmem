/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include <mutex>
#include "aclshmemi_global_state.h"
#include "host/aclshmem_host_def.h"
#include "utils/aclshmemi_host_types.h"
#include "utils/aclshmemi_logger.h"
#include "utils/utils.h"

#define LOAD_SYM(TARGET_FUNC, FILE_HANDLE, SYMBOL_NAME)                                         \
    dlerror();                                                                                  \
    *((void **)&TARGET_FUNC) = dlsym(FILE_HANDLE, SYMBOL_NAME);                                 \
    error = dlerror();                                                                          \
    if (error != NULL) {                                                                        \
        fprintf(stderr, "dlsym failed: %s\n", error);                                           \
        dlclose(hal_handle);                                                                    \
    }

std::mutex g_mutex;

bool g_hal_loaded = false;
static void *hal_handle;
const char *g_hal_lib_name = "libascend_hal.so";

int (*halMemAddressReserveFunc)(void **ptr, size_t size, size_t alignment, void *addr, uint64_t flag);
int (*halMemAddressFreeFunc)(void *ptr);
int (*halMemCreateFunc)(drv_mem_handle_t **handle, size_t size, const struct drv_mem_prop *prop, uint64_t flag);
int (*halMemReleaseFunc)(drv_mem_handle_t *handle);
int (*halMemMapFunc)(void *ptr, size_t size, size_t offset, drv_mem_handle_t *handle, uint64_t flag);
int (*halMemUnmapFunc)(void *ptr);

int32_t load_hal_library()
{
    char *error;
    std::lock_guard<std::mutex> guard(g_mutex);
    if (g_hal_loaded) {
        return 0;
    }
    
    dlerror();

    hal_handle = dlopen(g_hal_lib_name, RTLD_NOW);
    if (!hal_handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    LOAD_SYM(halMemAddressReserveFunc, hal_handle, "halMemAddressReserve");
    LOAD_SYM(halMemAddressFreeFunc, hal_handle, "halMemAddressFree");
    LOAD_SYM(halMemCreateFunc, hal_handle, "halMemCreate");
    LOAD_SYM(halMemReleaseFunc, hal_handle, "halMemRelease");
    LOAD_SYM(halMemMapFunc, hal_handle, "halMemMap");
    LOAD_SYM(halMemUnmapFunc, hal_handle, "halMemUnmap");

    g_hal_loaded = true;
    return 0;
}

global_state_reigister::global_state_reigister(int device_id): device_id_{device_id}
{
    ACLSHMEM_CHECK(load_hal_library());

    ACLSHMEM_CHECK(halMemAddressReserveFunc(&device_ptr_, GLOBAL_STATE_SIZE, 0, (void *)(SVM_END_ADDR - GLOBAL_STATE_SIZE), 1));

    int32_t logicDeviceId = -1;
    rtLibLoader& loader = rtLibLoader::getInstance();
    if (loader.isLoaded()) {
        loader.getLogicDevId(device_id_, &logicDeviceId);
    }

    drv_mem_prop memprop;
    memprop.side = 1;
    memprop.devid = logicDeviceId;
    memprop.module_id = 0;
    memprop.pg_type = 0;
    memprop.mem_type = 0;
    memprop.reserve = 0;

    ACLSHMEM_CHECK(halMemCreateFunc(&alloc_handle, GLOBAL_STATE_SIZE, &memprop, 0));

    ACLSHMEM_CHECK(halMemMapFunc(device_ptr_, GLOBAL_STATE_SIZE, 0, alloc_handle, 0));

    // init success
    init_status_ = 0;
}

global_state_reigister::~global_state_reigister()
{
    ACLSHMEM_CHECK(halMemUnmapFunc(device_ptr_));

    ACLSHMEM_CHECK(halMemReleaseFunc(alloc_handle));

    ACLSHMEM_CHECK(halMemAddressFreeFunc(device_ptr_));

    if (hal_handle != nullptr)
        dlclose(hal_handle);
}

void *global_state_reigister::get_ptr()
{
    return device_ptr_;
}

int global_state_reigister::get_init_status()
{
    return init_status_;
}
